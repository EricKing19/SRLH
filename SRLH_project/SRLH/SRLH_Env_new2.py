from PIL import Image
import numpy as np

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

def calc_rewards(query_codes, eval_codes, query_label, eval_label):
    """
    calc_rewards
    :param query_codes: N x k ,1,0
    :param eval_codes:
    :return:
    """
    top_ndcg = 100
    qB = 2 * query_codes - 1
    evalB = 2 * eval_codes - 1
    ndcg = cal_ndcg_k (qB, evalB, query_label.numpy (), eval_label.numpy (), top_ndcg)

    return ndcg

class Env:
    def __init__(self, z_scores, query_label, eval_index, sparsity, num_samples, database_labels, code_length):
        num_database = database_labels.shape[0]
        select_image = list (np.random.permutation (range (num_database)))[0: num_samples]
        sample_index = eval_index[select_image,:]
        sample_codes = np.zeros ([num_samples, code_length], dtype=int)
        sample_label = database_labels[select_image, :]
        candidate_set = [i for i in range (code_length)]
        query_code = np.zeros ([code_length], dtype=int)

        self.z_scores = z_scores
        self.query_label = query_label
        self.sparsity = sparsity
        self.sample_index = sample_index
        self.sample_label = sample_label
        self.sample_codes = sample_codes
        self.candidate_set = candidate_set
        self.query_code = query_code
        self.cur_index = []

    def encode_codes(self, action, sample_num, s):
        #query_x = 0
        query_y = action
        sample_x = range (sample_num)
        sample_y = self.sample_index[:, s]
        self.query_code[query_y] = 1
        self.sample_codes[sample_x, sample_y] = 1

    def step(self, action, cur_index):

        reward = calc_rewards (self.query_code, self.sample_codes, self.query_label, self.sample_label)

        del self.candidate_set[action]
        done = len(self.candidate_set) == 0

        self.cur_index = cur_index
        return reward, done

    def reset(self, data_dict, crop_size=224):
        n_train_img = len(data_dict['d_img_list'])
        img_list = []
        score_list = []
        for i in range(n_train_img):
            img_path = join(data_dict['base_path'], data_dict['d_img_list'][i])
            PILimg = Image.open(img_path)
            img_list.append(img_transform_tr(crop_size)(PILimg))
            score_list.append(data_dict['score_list'][i])

        ## whether a query image exists
        try:
            q_img_path = join(data_dict['base_path'], data_dict['q_img'])
            q_img = img_transform_tr(crop_size)(Image.open(q_img_path))
            q_score = data_dict['q_score']
        except Exception as e:
            q_img = None
            q_score = None

        return img_list, score_list, q_img, q_score

    def render(self, mode='human', close=False):
        raise NotImplementedError()

    def close(self):
        return

    def seed(self, seed=None):
        pass

    def __del__(self):
        self.close()

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)