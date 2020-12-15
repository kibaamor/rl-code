#!/usr/bin/python3
# coding=utf-8
import numpy as np


class SarsaAgent:
    def __init__(self, obs_dim, act_dim, lr=0.1, e_greed=0.1, gamma=0.95):
        self.Q_pi = np.zeros((obs_dim, act_dim))
        self.act_dim = act_dim
        self.lr = lr
        self.e_greed = e_greed
        self.gamma = gamma

    def predict(self, obs):
        q_list = self.Q_pi[obs, :]
        q_max = np.max(q_list)
        choice_list = np.where(q_list == q_max)[0]
        return np.random.choice(choice_list)

    def sample(self, obs):
        if np.random.rand() < self.e_greed:
            return np.random.choice(self.act_dim)
        else:
            return self.predict(obs)

    def learn(self, obs, act, reward, next_obs, next_act, done):
        predict = self.Q_pi[obs, act]
        td_target = reward + self.gamma * self.Q_pi[next_obs, next_act]
        td_error = predict - td_target
        self.Q_pi[obs, act] -= self.lr * td_error
