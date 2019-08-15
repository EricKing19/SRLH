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
import SRLH.SRLH_Agent as Agent
import utils.calc_hr as calc_hr
from utils.DC_load_data import load_dataset
import SRLH.SRLH_Env_new2 as Environment


parser = argparse.ArgumentParser (description="SRLH demo")
parser.add_argument ('--bits', default='16', type=str,
                     help='binary code length (default: 8,16,32,64)')
parser.add_argument ('--gpu', default='0', type=str,
                     help='selected gpu (default: 3)')
parser.add_argument ('--dataname', default='SUN20', type=str,
                     help='MirFlickr, NUSWIDE, COCO, CIFAR10, CIFAR100, Mnist, fashion_mnist, STL10, SUN20')
parser.add_argument ('--arch', default='vgg11', type=str,
                     help='model name (default: resnet50,vgg11)')

parser.add_argument ('--epochs', default=20, type=int,
                     help='number of epochs (default: 1)')
parser.add_argument ('--batch-size', default=48, type=int,
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


def get_dist_graph(all_points, num_anchor=300):
    """
    get the cluster center as anchor by K-means++
    and calculate distance graph (n data points vs m anchors),
    :param all_points: n data points
    :param num_anchor:  m anchors, default = 300
    :return: distance graph n X m
    """
    # kmeans = KMeans (n_clusters=num_anchor, random_state=0, n_jobs=16, max_iter=50).fit_transform(all_points)
    # print ('dist graph done!')
    # return np.asarray(kmeans)
    ## smaple

    num_data = np.size (all_points, 0)
    sample_rate = 3000
    # sample_rate = num_data
    ind = random.sample (range (num_data), sample_rate)
    sample_points = all_points[ind, :]
    kmeans = KMeans (n_clusters=num_anchor, random_state=0, n_jobs=16, max_iter=50).fit (sample_points)
    km = kmeans.transform (all_points)
    print ('dist graph done!')
    return np.asarray (km)


def calc_all_loss(B, F, Z, inv_A, Z1, Z2, Y1, Y2, rho1, rho2, lambda_1, lambda_2):
    """
    Calculate loss: Tr(BLF^t) = Tr(B * Z * inv_A * Z^t * F^t)
    :param F: output of network n X k
    :param B: binary codes k X n
    :param Z: anchor graph, n X m
    :return: loss: trace(BLF)
    """
    bit, num_train = B.shape
    Z_T = Z.transpose ()  # m X n
    temp = np.dot (B, np.dot (Z, inv_A))  # k X m
    temp2 = np.dot (temp, Z_T)  # k X n
    BAF = np.dot (temp2, F)  # k X k
    Tr_BLF = np.trace (np.dot (B, F) - BAF)

    nI_K = num_train * np.eye (bit, bit)
    res1 = B - Z1
    res2 = B - Z2

    reg_loss = (B - F.transpose ()) ** 2
    oth_loss = lambda_1 * ((np.dot (B, B.transpose ()) - nI_K) ** 2) / 4
    bla_loss = lambda_2 * (B.sum (1) ** 2) / 2
    z1_loss = rho1 * ((res1 ** 2).sum ()) / 2
    z2_loss = rho2 * ((res2 ** 2).sum ()) / 2
    y1_loss = np.trace (np.dot (res1, Y1.transpose ()))
    y2_loss = np.trace (np.dot (res2, Y2.transpose ()))

    loss = Tr_BLF + 0.5 * reg_loss.sum () + oth_loss.sum () + bla_loss.sum () + z1_loss + z2_loss + y1_loss + y2_loss

    #print ('Tr_BLF:' + str (Tr_BLF + 0.5 * reg_loss.sum ()))
    #print ('loss all done!')
    return loss


def encoding_onehot(target, nclasses=10):
    target_onehot = torch.FloatTensor (target.size (0), nclasses)
    target_onehot.zero_ ()
    target_onehot.scatter_ (1, target.view (-1, 1), 1)
    return target_onehot


# def _dataset(dataname):
#     # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     transformations = transforms.Compose([
#         transforms.Scale(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         normalize
#     ])
#
#     rootpath = os.path.join('/data/dacheng/Datasets/', dataname)
#
#     if dataname=='NUSWIDE':
#         dset_database = dataset.NUSWIDE ('train_img.txt', 'train_label.txt', transformations)
#         dset_test = dataset.NUSWIDE ('test_img.txt', 'test_label.txt', transformations)
#     elif dataname=='MirFlickr':
#         dset_database = dataset.MirFlickr ('train_img.txt', 'train_label.txt', transformations)
#         dset_test = dataset.MirFlickr ('test_img.txt', 'test_label.txt', transformations)
#     elif dataname =='COCO':
#         dset_database = dataset.COCO ('train_img.txt', 'train_label.txt', transformations)
#         dset_test = dataset.COCO ('test_img.txt', 'test_label.txt', transformations)
#     elif dataname == 'CIFAR10':
#         dset_database = dataset.CIFAR10 ('train_img.txt', 'train_label.txt', transformations)
#         dset_test = dataset.CIFAR10 ('test_img.txt', 'test_label.txt', transformations)
#     elif dataname == 'MNIST':
#         dset_database = dataset.MNIST (True, transformations)
#         dset_test = dataset.MNIST (False, transformations)
#
#     num_database, num_test = len (dset_database), len (dset_test)
#
#     def load_label(filename, DATA_DIR):
#         path = os.path.join(DATA_DIR, filename)
#         fp = open(path, 'r')
#         labels = [x.strip() for x in fp]
#         fp.close()
#         return torch.LongTensor(list(map(int, labels)))
#
#     def DC_load_label(filename, DATA_DIR):
#         path = os.path.join(DATA_DIR, filename)
#         label = np.loadtxt (path, dtype=np.int64)
#         return torch.LongTensor(label)
#
#     def load_label_CIFAR100(root, train=True):
#         base_folder = 'cifar-100-python'
#         train_list = [
#             ['train', '16019d7e3df5f24257cddd939b257f8d'],
#         ]
#
#         test_list = [
#             ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
#         ]
#
#         root = os.path.expanduser(root)
#         train = train  # training set or test set
#
#         # now load the picked numpy arrays
#         if train:
#             train_data = []
#             train_labels = []
#             for fentry in train_list:
#                 f = fentry[0]
#                 file = os.path.join(root, base_folder, f)
#                 fo = open(file, 'rb')
#                 if sys.version_info[0] == 2:
#                     entry = pickle.load(fo)
#                 else:
#                     entry = pickle.load(fo, encoding='latin1')
#                 train_data.append(entry['data'])
#                 if 'labels' in entry:
#                     train_labels += entry['labels']
#                 else:
#                     train_labels += entry['fine_labels']
#                 fo.close()
#
#         else:
#             f = test_list[0][0]
#             file = os.path.join(root, base_folder, f)
#             fo = open(file, 'rb')
#             if sys.version_info[0] == 2:
#                 entry = pickle.load(fo)
#             else:
#                 entry = pickle.load(fo, encoding='latin1')
#             test_data = entry['data']
#             if 'labels' in entry:
#                 test_labels = entry['labels']
#             else:
#                 test_labels = entry['fine_labels']
#             fo.close()
#
#         if train:
#             target = train_labels
#         else:
#             target = test_labels
#         return torch.LongTensor(list(map(int, target)))
#
#
#     def DC_load_label_MNIST(filename, root):
#         _, labels = torch.load (os.path.join (root, filename))
#         return torch.LongTensor(labels)
#
#     if dataname=='CIFAR10':
#         testlabels_ = load_label('test_label.txt', rootpath)
#         databaselabels_ = load_label('train_label.txt', rootpath)
#         testlabels = encoding_onehot(testlabels_)
#         databaselabels = encoding_onehot(databaselabels_)
#     elif dataname == 'MNIST':
#         databaselabels_ = DC_load_label_MNIST ('training.pt', root='/home/dacheng/PycharmProjects/ADSH_pytorch/data/processed/')
#         testlabels_ = DC_load_label_MNIST ('test.pt', root='/home/dacheng/PycharmProjects/ADSH_pytorch/data/processed/')
#         testlabels = encoding_onehot(testlabels_)
#         databaselabels = encoding_onehot(databaselabels_)
#     elif dataname == 'CIFAR100':
#         testlabels_ = load_label_CIFAR100('/data/dacheng/Datasets/CIFAR100/', train=False)
#         databaselabels_ = load_label_CIFAR100('/data/dacheng/Datasets/CIFAR100/', train=True)
#         testlabels = encoding_onehot(testlabels_, nclasses=100)
#         databaselabels = encoding_onehot(databaselabels_, nclasses=100)
#     else:
#         databaselabels = DC_load_label('train_label.txt', rootpath)
#         testlabels = DC_load_label('test_label.txt', rootpath)
#
#     dsets = (dset_database, dset_test)
#     nums = (num_database, num_test)
#     labels = (databaselabels, testlabels)
#
#     return nums, dsets, labels

def calc_loss(B, F, Z, inv_A, lambda_1, lambda_2, lambda_3, code_length):
    """
    Calculate loss: Tr(BLF^t) = Tr(B * Z * inv_A * Z^t * F^t)
    :param F: output of network n X k
    :param B: binary codes k X n
    :param Z: anchor graph, n X m
    :return: loss: trace(BLF)
    """
    Z_T = Z.transpose ()  # m X n
    temp = np.dot (B, np.dot (Z, inv_A))  # k X m
    temp2 = np.dot (temp, Z_T)  # k X n
    BAF = np.dot (temp2, F)  # k X k
    Tr_BLF = np.trace (np.dot (B, F) - BAF)

    num_train = np.size (B, 1)
    # nI_K =  num_train * np.eye (code_length, code_length)
    nI_K = np.eye (code_length, code_length)

    one_vectors = np.ones ((num_train, code_length))
    reg_loss = (B - F.transpose ()) ** 2
    # oth_loss = (np.dot(F.transpose(), F) - nI_K) ** 2
    oth_loss = (np.dot (F.transpose (), F) / num_train - nI_K) ** 2

    mean_F = F.mean (0).reshape (1, code_length)
    var_loss = (F - mean_F) ** 2

    bla_loss = (F.sum (0)) ** 2
    susb_loss = np.abs (F) - one_vectors
    L1_loss = np.abs (susb_loss)


    loss = (Tr_BLF + 0.5 * (reg_loss.sum () + lambda_2 * bla_loss.sum () - lambda_2 * var_loss.sum () + lambda_3 * L1_loss.sum ())) / (code_length * num_train) + 0.25 * lambda_1 * oth_loss.sum () / (code_length)

    print ('Tr_BLF:' + str (Tr_BLF / code_length / num_train))
    print ('reg_loss:' + str (reg_loss.sum () / code_length / num_train))
    print ('oth_loss:' + str (lambda_1 * oth_loss.sum () / code_length))
    print ('bla_loss:' + str (lambda_2 * bla_loss.sum () / code_length / num_train))
    print ('L1_loss:' + str (lambda_3 * L1_loss.sum () / code_length / num_train))
    print ('var_loss:' + str (lambda_2 * var_loss.sum () / code_length / num_train))
    print ('loss done!')
    return loss

def calc_R1_loss(B, F, Z, inv_A, lambda_1, lambda_2, lambda_3, beta_1, beta_2, code_length):
    """
    Calculate loss: Tr(BLF^t) = Tr(B * Z * inv_A * Z^t * F^t)
    :param F: output of network n X k
    :param B: binary codes k X n
    :param Z: anchor graph, n X m
    :return: loss: trace(BLF)
    """
    Z_T = Z.transpose ()  # m X n
    temp = np.dot (B, np.dot (Z, inv_A))  # k X m
    temp2 = np.dot (temp, Z_T)  # k X n
    BAF = np.dot (temp2, F)  # k X k
    Tr_BLF = np.trace (np.dot (B, F) - BAF)

    num_train = np.size (B, 1)
    # nI_K =  num_train * np.eye (code_length, code_length)
    nI_K = np.eye (code_length, code_length)

    one_vectors = np.ones ((num_train, code_length))
    reg_loss = (B - F.transpose ()) ** 2
    # oth_loss = (np.dot(F.transpose(), F) - nI_K) ** 2
    oth_loss = (np.dot (F.transpose (), F) / num_train - nI_K) ** 2

    mean_F = F.mean (0).reshape (1, code_length)
    var_loss = (F - mean_F) ** 2

    bla_loss = (F.sum (0)) ** 2
    susb_loss = np.abs (F) - one_vectors
    L1_loss = np.abs (susb_loss)

    loss1 = (Tr_BLF + 0.5 * (reg_loss.sum () + lambda_2 * bla_loss.sum () + lambda_3 * L1_loss.sum ())) / (code_length * num_train) + 0.25 * lambda_1 * oth_loss.sum () / (code_length)
    othB_loss = beta_1 * ((np.dot (B, B.transpose ()) - nI_K) ** 2) / 4
    blaB_loss = beta_2 * (B.sum (1) ** 2) / 2
    loss2 = othB_loss.sum () / (code_length) + blaB_loss.sum () / (code_length * num_train)

    print ('Tr_BLF:' + str (Tr_BLF / code_length / num_train))
    print ('reg_loss:' + str (reg_loss.sum () / code_length / num_train))
    print ('oth_loss:' + str (lambda_1 * oth_loss.sum () / code_length))
    print ('bla_loss:' + str (lambda_2 * bla_loss.sum () / code_length / num_train))
    print ('L1_loss:' + str (lambda_3 * L1_loss.sum () / code_length / num_train))
    print ('var_loss:' + str (lambda_2 * var_loss.sum () / code_length / num_train))
    print ('loss done!')


    return loss1 + loss2

def encode(model, data_loader, num_data, bit):
    B = np.zeros ([num_data, bit], dtype=np.float32)
    for iter, data in enumerate (data_loader, 0):
        data_input, _, data_ind = data
        data_input = Variable (data_input.cuda (), volatile=True)
        output = model (data_input)
        B[data_ind.numpy (), :] = torch.sign (output[1].cpu ().data).numpy ()
    return B


def get_F(model, data_loader, num_data, bit):
    B = np.zeros ([num_data, bit], dtype=np.float32)
    for iter, data in enumerate (data_loader, 0):
        data_input, _, data_ind = data
        data_input = Variable (data_input.cuda (), volatile=True)
        output = model (data_input)
        B[data_ind.numpy (), :] = output[1].cpu ().data.numpy ()
    return B


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

def adjusting_learning_rate2(optimizer, ite, epo):
    cur_epoch = 2 * ite + epo
    epoch_all = opt.max_iter * opt.epochs
    lr = opt.learning_rate*((1- float(cur_epoch) /epoch_all)**0.9)
    optimizer.param_groups[0]['lr'] = lr * 10
    optimizer.param_groups[1]['lr'] = lr
    print ('learning rate is adjusted!')


def SRLH_algo(code_length, dataname):
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
    model.cuda ()
    cudnn.benchmark = True

    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = optim.SGD (model.parameters (), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)  ####
    # optimizer = optim.RMSprop (model.parameters (), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = optim.RMSprop (model.parameters (), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    # optimizer = optim.Adadelta (model.parameters (), weight_decay=weight_decay)
    # optimizer = optim.Adam (model.parameters ())

    hash_params = list (map (id, model.hash.parameters ()))
    base_params = filter (lambda p: id (p) not in hash_params, model.parameters ())
    params = [
        {"params": model.hash.parameters(), "lr": learning_rate * 10},
        {"params": base_params, "lr": learning_rate},
    ]
    optimizer = optim.SGD (params, weight_decay=weight_decay, momentum=0.9)
    #optim.Adam (params)

    evalmodel_PATH = '/data/dacheng/SRLH/model.pkl'
    torch.save (model, evalmodel_PATH)

    '''
    learning deep neural network: feature learning
    '''
    trainloader = DataLoader (dset_database, batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)

    agent = Agent.Agent(eps, EPSILON)
    model.train ()
    for epoch in range (epochs):
        iter_time = time.time ()
        eval_model = torch.load (evalmodel_PATH)
        eval_index = get_index (eval_model, trainloader, num_database, code_length)

        # adjusting_learning_rate2 (optimizer, iter, epoch)

        print (optimizer.param_groups[0]['lr'])
        print (optimizer.param_groups[1]['lr'])

        running_loss = 0

        for iteration, (train_input, train_label, batch_ind) in enumerate (trainloader):
            train_input = Variable (train_input.cuda ())
            z_scores = model (train_input)

            optimizer.zero_grad ()
            for batch_i in range (len(batch_ind)):
                per_env = Environment.Env (z_scores[batch_i,:], train_label[batch_i,:], eval_index, sparsity, num_samples, database_labels, code_length)
                # running_loss = 0
                actions, log_probs, rewards = agent.sample_episode (per_env)
                # print log_probs
                # print rewards
                policy_loss = agent.cal_loss(log_probs, rewards, gamma)

                policy_loss = torch.cat (policy_loss).sum () / batch_size
                running_loss += policy_loss.data[0]
                policy_loss.backward (retain_graph=True)
            optimizer.step ()

            # print('[%d, %5d] loss: %.5f' % (epoch + 1, iteration + 1, running_loss ))

            if (iteration % 50) == 49:
                # print ('iteration:' + str (iteration))
                print('[%d, %5d] loss: %.5f' %
                      (epoch + 1, iteration + 1, running_loss / 50))
                print   rewards 
                running_loss = 0.0

        torch.save (model, evalmodel_PATH)

    # adjusting_learning_rate (optimizer, iter)

    iter_time = time.time () - iter_time


    # logger.info ('[Iteration: %3d/%3d][Train Loss: before:%.4f, after:%.4f]', iter, max_iter, loss_before, loss_)
    # record['train loss'].append (loss_)
    record['iter time'].append (iter_time)


    '''
    training procedure finishes, evaluation
    '''
    model.eval ()
    retrievalloader = DataLoader (dset_database, batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=4)
    testloader = DataLoader (dset_test, batch_size=batch_size,
                             shuffle=False,
                             num_workers=4)
    qB = get_codes (model, testloader, num_test, code_length, sparsity)
    rB_sy = get_codes (model, retrievalloader, num_database, code_length, sparsity)


    topKs = np.arange (1, 500, 50)
    top_ndcg = 100

    # Pres_asy = calc_hr.calc_topk_pres (qB, rB_asy, test_labels.numpy (), database_labels.numpy (), topKs)
    # ndcg_asy = calc_hr.cal_ndcg_k (qB, rB_asy, test_labels.numpy (), database_labels.numpy (), top_ndcg)
    Pres_sy = calc_hr.calc_topk_pres (qB, rB_sy, test_labels.numpy (), database_labels.numpy (), topKs)
    ndcg_sy = calc_hr.cal_ndcg_k (qB, rB_sy, test_labels.numpy (), database_labels.numpy (), top_ndcg)

    map_sy = calc_hr.calc_map (qB, rB_sy, test_labels.numpy (), database_labels.numpy ())
    # map_asy = calc_hr.calc_map (qB, rB_asy, test_labels.numpy (), database_labels.numpy ())
    top_map_sy = calc_hr.calc_topMap (qB, rB_sy, test_labels.numpy (), database_labels.numpy (), 2000)
    # top_map_asy = calc_hr.calc_topMap (qB, rB_asy, test_labels.numpy (), database_labels.numpy (), 2000)

    logger.info ('[Evaluation: mAP_sy: %.4f]', map_sy)
    # logger.info ('[Evaluation: mAP_asy: %.4f]', map_asy)
    logger.info ('[Evaluation: topK_mAP_sy: %.4f]', top_map_sy)
    # logger.info ('[Evaluation: topK_mAP_asy: %.4f]', top_map_asy)
    logger.info ('[Evaluation: Pres_sy: %.4f]', Pres_sy[0])
    print Pres_sy
    # logger.info ('[Evaluation: Pres_asy: %.4f]', Pres_asy[0])
    # print Pres_asy
    logger.info ('[Evaluation: topK_ndcg_sy: %.4f]', ndcg_sy)
    # logger.info ('[Evaluation: topK_ndcg_asy: %.4f]', ndcg_asy)
    record['rB_sy'] = rB_sy

    record['qB'] = qB
    record['map_sy'] = map_sy
    # record['map_asy'] = map_asy
    record['topK_map_sy'] = top_map_sy
    # record['topK_map_asy'] = top_map_asy
    record['topK_ndcg_sy'] = ndcg_sy
    # record['topK_ndcg_asy'] = ndcg_asy
    record['Pres_sy'] = Pres_sy
    # record['Pres_asy'] = Pres_asy
    filename = os.path.join (logdir, str (code_length) + 'bits-record.pkl')

    _save_record (record, filename)
    return top_map_sy


if __name__ == "__main__":
    global opt, logdir
    opt = parser.parse_args ()
    logdir = '-'.join (['log/run', opt.dataname, datetime.now ().strftime ("%y-%m-%d-%H-%M-%S")])
    _logging ()
    _record ()
    bits = [int (bit) for bit in opt.bits.split (',')]
    for bit in bits:
        SRLH_algo (bit, opt.dataname)
    # lambda_1s = [ float(lambda_1) for lambda_1 in opt.lambda_1.split (',')]
    # lambda_2s = [int (lambda_2) for lambda_2 in opt.lambda_2.split (',')]
    # lambda_3s = [float (lambda_3) for lambda_3 in opt.lambda_3.split (',')]
    # topmap_lambda = np.zeros ((len(lambda_1s),len(lambda_3s)))
    # for i, lambda_1 in enumerate(lambda_1s):
    #     for j, lambda_3 in enumerate(lambda_3s):
    #
    #         topmap = DAGH_algo(lambda_1,lambda_3,opt.dataname)
    #
    #         topmap_lambda[i,j] = topmap
    # record['topmap_lambda'] = topmap_lambda
