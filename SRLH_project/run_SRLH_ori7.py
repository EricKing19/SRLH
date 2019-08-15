import pickle
import os
import argparse
import logging
import torch
import time
import sys
import random

import numpy as np
from numpy.linalg import *
import torch.optim as optim
import torchvision.transforms as transforms

from sklearn.cluster import KMeans
from datetime import datetime
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import SRLH.SRLH_Policy as PolicyNet
import SRLH.SRLH_Agent_sigmoid2 as Agent
import utils.calc_hr as calc_hr
from utils.DC_load_data import load_dataset
import SRLH.SRLH_Env_random as Environment


parser = argparse.ArgumentParser (description="SRLH demo sigmoid2 error")
parser.add_argument ('--bits', default='64', type=str,
                     help='binary code length (default: 8,16,32,64)')
parser.add_argument ('--gpu', default='1', type=str,
                     help='selected gpu (default: 3)')
parser.add_argument ('--dataname', default='SUN20', type=str,
                     help='MirFlickr, NUSWIDE, COCO, CIFAR10, CIFAR100, Mnist, fashion_mnist, STL10, SUN20')
parser.add_argument ('--arch', default='vgg11', type=str,
                     help='model name (default: resnet50,vgg11)')
parser.add_argument ('--optimizer', default='RMSprop', type=str,
                     help='optimizer name (default: SGD, Adam)')

parser.add_argument ('--epochs', default=20, type=int,
                     help='number of epochs (default: 1)')
parser.add_argument ('--batch-size', default=2, type=int,
                     help='batch size (default: 64)')
parser.add_argument ('--learning-rate', default=1e-4, type=float,
                     help='hyper-parameter: learning rate (default: 10**-3)')

parser.add_argument('--num-samples', default=1000, type=int,
                    help='hyper-parameter: number of samples (default: 2000)')
parser.add_argument('--sparsity', default=10, type=int,
                    help='hyper-parameter: number of sparsity (default: 10)')
parser.add_argument('--gamma', default=1, type=float,
                    help='hyper-parameter: discount factor (default: 0.2)')
parser.add_argument('--EPSILON', type=float, default=0.,
                    help='EPSILON greedy policy (default: 0.)')


def _logging():
    os.mkdir (logdir)
    global logger
    logfile = os.path.join (logdir, 'log.log')
    logger = logging.getLogger ('')
    logger.setLevel (logging.INFO)
    fh = logging.FileHandler (logfile)
    fh.setLevel (logging.INFO)
    ch = logging.StreamHandler ()
    ch.setLevel (logging.INFO)

    _format = logging.Formatter ("%(name)-4s: %(levelname)-4s: %(message)s")
    fh.setFormatter (_format)
    ch.setFormatter (_format)

    logger.addHandler (fh)
    logger.addHandler (ch)
    return


def _record():
    global record
    record = {}
    record['train loss'] = []
    record['iter time'] = []
    record['param'] = {}
    record['stime'] = []
    record['btime'] = []
    record['ntime'] = []
    return


def _save_record(record, filename):
    with open (filename, 'wb') as fp:
        pickle.dump (record, fp)
    return

def get_codes(model, data_loader, num_data, bit, sparsity):
    codes = np.zeros ([num_data, bit], dtype=np.float32)
    for iter, (data_input, _, data_ind) in enumerate (data_loader, 0):
        data_input = Variable (data_input.cuda (), volatile=True)
        output = model (data_input)
        codes[data_ind.numpy (), :] = output.cpu ().data.numpy ()
    ind = np.argsort (-codes)

    sparese_codes = np.zeros ([num_data, bit], dtype=int)

    for i in range(num_data):
        x = i * np.ones (sparsity, dtype=int)
        y = ind[i, 0:sparsity]
        sparese_codes[x, y] = 1

    binary_codes = sparese_codes * 2 - 1
    return binary_codes


def get_index(model, data_loader, num_data, bit):
    codes = np.zeros ([num_data, bit], dtype=np.float32)
    for iter, (data_input, _, data_ind) in enumerate (data_loader, 0):
        data_input = Variable (data_input.cuda (), volatile=True)
        output = model (data_input)
        codes[data_ind.numpy (), :] = output.cpu ().data.numpy ()
    ind = np.argsort (-codes)

    return ind

def adjusting_learning_rate(optimizer, iter):
    if ((iter % 2) == 0) & (iter !=0):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 5
        print ('learning rate is adjusted!')

def adjusting_learning_rate2(optimizer, epo):
    epoch_all = opt.epochs
    lr = opt.learning_rate*((1- float(epo) /epoch_all)**0.9)
    optimizer.param_groups[0]['lr'] = lr * 10
    optimizer.param_groups[1]['lr'] = lr
    print ('learning rate is adjusted!')


def test(model, nums, dsets, labels, code_length):
    model.eval()
    num_database, num_test = nums
    dset_database, dset_test = dsets
    database_labels, test_labels = labels

    retrievalloader = DataLoader(dset_database, batch_size=opt.batch_size,
                                 shuffle=False,
                                 num_workers=4)

    testloader = DataLoader(dset_test, batch_size=opt.batch_size,
                            shuffle=False,
                            num_workers=4)
    qB = get_codes(model, testloader, num_test, code_length, opt.sparsity)
    rB_sy = get_codes(model, retrievalloader, num_database, code_length, opt.sparsity)

    topKs = np.arange(1, 500, 50)
    top_ndcg = 100

    Pres_sy = calc_hr.calc_topk_pres(qB, rB_sy, test_labels.numpy(), database_labels.numpy(), topKs)
    ndcg_sy = calc_hr.cal_ndcg_k(qB, rB_sy, test_labels.numpy(), database_labels.numpy(), top_ndcg)

    map_sy = calc_hr.calc_map(qB, rB_sy, test_labels.numpy(), database_labels.numpy())
    top_map_sy = calc_hr.calc_topMap(qB, rB_sy, test_labels.numpy(), database_labels.numpy(), 2000)

    logger.info ('[Evaluation: mAP_sy: %.4f]', map_sy)
    logger.info ('[Evaluation: topK_mAP_sy: %.4f]', top_map_sy)
    logger.info ('[Evaluation: Pres_sy: %.4f]', Pres_sy[0])
    print Pres_sy
    logger.info ('[Test Evaluation: topK_ndcg_sy: %.4f]', ndcg_sy)

    return qB, rB_sy, map_sy, top_map_sy, Pres_sy, ndcg_sy


def SRLH_algo(code_length, dataname, date):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    torch.manual_seed (0)
    torch.cuda.manual_seed (0)
    # code_length=8

    '''
    parameter setting
    '''
    epochs = opt.epochs
    batch_size = opt.batch_size
    learning_rate = opt.learning_rate
    weight_decay = 5 * 10 ** -4

    num_samples = opt.num_samples
    sparsity = opt.sparsity
    gamma = opt.gamma
    EPSILON = opt.EPSILON
    optimizer_name = opt.optimizer
    eps = np.finfo (np.float32).eps.item ()

    record['param']['opt'] = opt
    record['param']['description'] = '[Comment: learning rate decay]'
    logger.info (opt)
    logger.info (code_length)
    logger.info (record['param']['description'])

    '''
    dataset preprocessing
    '''
    nums, dsets, labels = load_dataset (dataname, False, 20)
    num_database, num_test = nums
    dset_database, dset_test = dsets
    database_labels, test_labels = labels

    '''
    model construction
    '''
    model = PolicyNet.Policy_net (opt.arch, code_length)
    model = torch.nn.DataParallel(model)
    model.cuda ()
    cudnn.benchmark = True

    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = optim.SGD (model.parameters (), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)  ####
    # optimizer = optim.RMSprop (model.parameters (), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = optim.RMSprop (model.parameters (), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    # optimizer = optim.Adadelta (model.parameters (), weight_decay=weight_decay)
    # optimizer = optim.Adam (model.parameters ())

    hash_params = list (map (id, model.module.hash.parameters ()))
    base_params = filter (lambda p: id (p) not in hash_params, model.module.parameters ())
    params = [
        {"params": model.module.hash.parameters(), "lr": learning_rate},
        {"params": base_params, "lr": learning_rate / 10},
    ]

    if optimizer_name == 'SGD':
        optimizer = optim.SGD (params, weight_decay=weight_decay, momentum=0.9)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam (params, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop (params, weight_decay=weight_decay)
    else:
        raise ValueError('no such optimizer')


    modeldir = '-'.join (['/data/dacheng/SRLH/', opt.dataname, date])
    os.mkdir(modeldir)
    evalmodel_PATH = modeldir + '/model.pkl'
    torch.save (model, evalmodel_PATH)

    '''
    learning deep neural network: feature learning
    '''
    trainloader = DataLoader (dset_database, batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)

    agent = Agent.Agent(eps, EPSILON)
    for epoch in range (epochs):
        model.train()
        print(' ============================= Training ===================================')

        iter_time = time.time ()
        eval_model = torch.load (evalmodel_PATH)
        eval_index = get_index (eval_model, trainloader, num_database, code_length)

        if optimizer_name == 'SGD':
            adjusting_learning_rate2 (optimizer, epoch)

        print (optimizer.param_groups[0]['lr'])
        print (optimizer.param_groups[1]['lr'])

        running_loss = 0
        rewards_temp = np.zeros(sparsity)
        log_probs_temp = np.zeros(sparsity)

        for iteration, (train_input, train_label, batch_ind) in enumerate (trainloader):
            train_input = Variable (train_input.cuda ())
            z_scores = model (train_input)

            optimizer.zero_grad ()

            for batch_i in range (len(batch_ind)):
                per_env = Environment.Env (z_scores[batch_i,:], train_label[batch_i,:], eval_index, sparsity, num_samples, database_labels, code_length)
                actions, log_probs, rewards = agent.sample_episode (per_env)

                policy_loss = agent.cal_loss(log_probs, rewards, gamma)

                policy_loss = torch.cat (policy_loss).sum () / batch_size
                running_loss += policy_loss.data[0]
                policy_loss.backward (retain_graph=True)

                rewards_temp += np.asarray(rewards) / len (batch_ind)
                log_probs_temp += np.asarray([r.cpu().data for r in log_probs]) / len(batch_ind)

            optimizer.step ()


            if (iteration % 20) == 19:
                print('[%d, %5d] loss: %.5f' %
                      (epoch + 1, iteration + 1, running_loss / 20))
                print (rewards_temp / 20 )
                print (log_probs_temp / 20)

                rewards_temp = np.zeros(opt.sparsity)
                log_probs_temp = np.zeros(sparsity)
                running_loss = 0.0

        torch.save (model, evalmodel_PATH)

        if (epoch % 2) == 1:
            print(' ============================= Testing ===================================')
            qB, rB_sy, map_sy, top_map_sy, Pres_sy, ndcg_sy = test(model, nums, dsets, labels, code_length)


    iter_time = time.time () - iter_time

    record['iter time'].append (iter_time)
    record['rB_sy'] = rB_sy
    record['qB'] = qB
    record['map_sy'] = map_sy
    record['topK_map_sy'] = top_map_sy
    record['topK_ndcg_sy'] = ndcg_sy
    record['Pres_sy'] = Pres_sy
    filename = os.path.join (logdir, str (code_length) + 'bits-record.pkl')
    _save_record (record, filename)


if __name__ == "__main__":
    global opt, logdir
    opt = parser.parse_args ()
    date = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    logdir = '-'.join (['log/run_ori7', opt.dataname, date])
    _logging ()
    _record ()
    bits = [int (bit) for bit in opt.bits.split (',')]
    for bit in bits:
        SRLH_algo (bit, opt.dataname, date)

