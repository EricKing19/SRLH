from PIL import Image
import numpy as np
import random

def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def cal_ndcg_k(qB, rB, queryL, retrievalL, topks):
    idx = np.arange (topks) + 1
    gnd = (np.dot(queryL, retrievalL.transpose())).astype(np.float32)
    hamm = calc_hammingDist(qB, rB)
    # ind = np.argsort(hamm)
    aaa = np.arange(0, retrievalL.shape[0])
    ind = np.lexsort((aaa, hamm))
    gnd = gnd[ind]
    tgnd = gnd[0:topks]
    a = ((2**tgnd - 1) / np.log (1 + idx)).sum()

    ind2 = np.argsort(-gnd)
    tgnd2 = gnd[ind2[0:topks]]
    b = ((2**tgnd2 - 1) / np.log (1 + idx)).sum ()

    ndcg =  a / b

    return ndcg

def calc_rewards(query_codes, eval_codes, query_label, eval_label, train_ndcg):
    """
    calc_rewards
    :param query_codes: N x k, 1,0
    :param eval_codes:
    :return:
    """
    qB = 2 * query_codes - 1
    evalB = 2 * eval_codes - 1
    ndcg = cal_ndcg_k (qB, evalB, query_label.numpy (), eval_label.numpy (), train_ndcg)

    return ndcg

class Env:
    def __init__(self, z_scores, query_label, eval_index, sparsity, num_samples, database_labels, code_length, train_ndcg):
        num_database = database_labels.shape[0]
        random_list = range(num_database)
        candidate_set = [i for i in range (code_length)]
        query_code = np.zeros ([code_length], dtype=int)

        self.code_length = code_length
        self.database_labels = database_labels
        self.num_samples = num_samples
        self.eval_index = eval_index
        self.random_list = random_list
        self.z_scores = z_scores
        self.query_label = query_label
        self.sparsity = sparsity
        self.candidate_set = candidate_set
        self.query_code = query_code
        self.cur_index = []
        self.train_ndcg = train_ndcg

    def encode_codes(self, action, s):
        #query_x = 0
        query_y = action
        self.query_code[query_y] = 1

        ### update the sampled data random
        select_image = random.sample(self.random_list, self.num_samples)
        sample_index = self.eval_index[select_image, :]
        sample_codes = np.zeros([self.num_samples, self.code_length], dtype=int)


        for i in range(self.num_samples):
            x = i * np.ones(s+1, dtype=int)
            y = sample_index[i, 0:s+1]
            sample_codes[x, y] = 1


        self.sample_codes = sample_codes
        self.sample_label = self.database_labels[select_image, :]

    def step(self, action, cur_index):

        reward = calc_rewards (self.query_code, self.sample_codes, self.query_label, self.sample_label, self.train_ndcg)

        del self.candidate_set[action]
        done = len(self.candidate_set) == 0

        self.cur_index = cur_index
        return reward, done

