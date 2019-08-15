import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from torch.nn import Module
from torch.distributions import Categorical
from scipy.stats import pearsonr, spearmanr
from torch.autograd import Variable



class Agent:
    def __init__(self, eps, EPSILON):
        self.eps = eps
        self.EPSILON = EPSILON

    def sel_sparsity(self, z_scores, candidate_set):
        return [z_scores[i] for i in candidate_set]

    def softmax_index(self, z_scores, candidate_set):
        sel_outputs = self.sel_sparsity (z_scores, candidate_set)
        return F.softmax (torch.cat (sel_outputs).view (-1), dim=0)


    def sel_action(self, z_scores, candidate_set):
        probs = self.softmax_index (z_scores, candidate_set)
        m = Categorical(probs)
        if np.random.uniform() < self.EPSILON:  # greedy
            action_temp = np.argmax(probs.data)*torch.ones(1,)
            action = Variable(action_temp.long()).cuda()
            log_prob = m.log_prob(action).view(1)
        else:  # random
            action = m.sample()
            log_prob = m.log_prob(action).view(1)
        return int(action.cpu ().data.numpy ()), log_prob

    def sample_episode(self, env):

        log_probs = []
        rewards = []
        actions = []
        for s in range(env.sparsity):
            action, log_prob = self.sel_action(env.z_scores, env.candidate_set)
            env.encode_codes (action, s)
            actions.append (env.candidate_set[action])
            reward, done = env.step (action, actions, s)

            log_probs.append (log_prob)
            rewards.append (reward)

            if done:
                break

        return actions, log_probs, rewards

    def cal_loss(self, log_probs, rewards, gamma):
        R = 0
        policy_loss = []
        returns = []
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert (0, R)
        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        returns = Variable (torch.from_numpy (returns).type (torch.FloatTensor).cuda ())

        for log_prob, reward in zip (log_probs, returns):
            policy_loss.append (-log_prob * reward)

        return policy_loss