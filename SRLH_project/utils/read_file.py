import pickle
import os
import torch
import numpy as np
import scipy.io
import utils.calc_hr as calc_hr
from utils.DC_load_data import load_dataset

def load_label(filename, DATA_DIR):
    path = os.path.join (DATA_DIR, filename)
    fp = open (path, 'r')
    labels = [x.strip () for x in fp]
    fp.close ()
    return torch.LongTensor (list (map (int, labels)))

# filename = '/home/dacheng/PycharmProjects/SRLH_project/log/run-SUN20-18-12-10-14-58-03/64bits-record.pkl'
# filename = '/home/dacheng/PycharmProjects/SRLH_project/log/run_ori2-SUN20-18-12-13-10-35-53/64bits-record.pkl'
# filename = '/home/dacheng/PycharmProjects/SRLH_project/log/run_ori4-SUN20-18-12-16-14-19-00/64bits-record.pkl'
# filename = '/home/dacheng/PycharmProjects/SRLH_project/log/run_ori5-SUN20-19-01-14-17-25-54/64bits-record.pkl'
filename = '/home/dacheng/PycharmProjects/SRLH_project/log/run_ori5-SUN20-19-01-16-11-34-42/64bits-record.pkl'

inf = pickle.load (open (filename))
qB =inf['qB']
rB_sy = inf['rB_sy']

ind_qb =inf['ind_qb']
ind_rb = inf['ind_rb']
#qB2 = 0.5 * (qB + 1)
#rB_sy2 = 0.5 * (rB_sy + 1)

dataname='SUN20'
nums, dsets, labels = load_dataset (dataname)
num_database, num_test = nums
dset_database, dset_test = dsets
database_labels, test_labels = labels

topKs = np.arange (1, 500, 50)
top_ndcg = 100

Pres_sy = calc_hr.calc_topk_pres (qB, rB_sy, test_labels.numpy (), database_labels.numpy (), topKs)
ndcg_sy = calc_hr.cal_ndcg_k (qB, rB_sy, test_labels.numpy (), database_labels.numpy (), top_ndcg)

map_sy = calc_hr.calc_map (qB, rB_sy, test_labels.numpy (), database_labels.numpy ())
top_map_sy = calc_hr.calc_topMap (qB, rB_sy, test_labels.numpy (), database_labels.numpy (), 2000)


scipy.io.savemat('SUN_64_data5.mat',{'qb':inf['qB'], 'b_sy':inf['rB_sy'],'ind_qb':inf['ind_qb'], 'ind_rb':inf['ind_rb']})
#
#scipy.io.savemat('Z_mean_B.mat',{'b_asy':inf['rB_asy'], 'b_sy':inf['rB_sy']})
dataname = 'CIFAR100'
rootpath = os.path.join ('/data/dacheng/Datasets/', dataname)

testlabels_ = load_label ('test_label.txt', rootpath).numpy()
databaselabels_ = load_label ('train_label.txt', rootpath).numpy()
code_length  = np.size (inf['rB'], 1)

# qB = np.zeros ((len(testlabels_), code_length), dtype=np.float)
# rB = np.zeros ((len(databaselabels_), code_length), dtype=np.float)
qB =[]
rB=[]

for i in range(100):
    qb_temp = inf['qB'][testlabels_==i]
    rb_temp = inf['rB'][databaselabels_==i]
    qB.append(qb_temp)
    rB.append(rb_temp)

qB =  np.array(qB).reshape((-1,8))
rB =  np.array(rB).reshape((-1,8))

inf
