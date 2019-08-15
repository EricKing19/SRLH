import numpy as np
import calc_hr as calc_hr

def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def cal_ndcg_k(qB, rB, queryL, retrievalL, topks):
    num_query = queryL.shape[0]
    idx = np.arange (topks) + 1
    ndcg = np.zeros([1,num_query], dtype=np.float32)
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose())).astype(np.float32)
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        tgnd = gnd[0:topks]
        a = ((2**tgnd - 1) / np.log (1 + idx)).sum()

        ind2 = np.argsort(-gnd)
        tgnd2 = gnd[ind2[0:topks]]
        b = ((2**tgnd2 - 1) / np.log (1 + idx)).sum ()

        ndcg[iter] =  a / b

        if (b == 0):
            print (iter)

    return ndcg

def calc_rewards(query_codes, eval_codes, query_label, eval_label):
    """
    calc_rewards
    :param query_codes: N x k ,1,0
    :param eval_codes:
    :return:
    """
    top_ndcg = eval_label.shape[0]
    qB = 2 * query_codes - 1
    evalB = 2 * eval_codes - 1
    ndcg = cal_ndcg_k (qB, evalB, query_label.numpy (), eval_label.numpy (), top_ndcg)

    return ndcg