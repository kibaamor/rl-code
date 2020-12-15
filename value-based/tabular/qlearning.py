#!/usr/bin/python3
# coding=utf-8
import numpy as np


class QLearningAgent:
    def __init__(self, obs_dim, act_dim, lr=0.1, e_greed=0.1, gamma=0.95):
        self.Q_star = np.zeros((obs_dim, act_dim))
        self.act_dim = act_dim
        self.lr = lr
        self.e_greed = e_greed
        self.gamma = gamma

    def predict(self, obs):
        q_list = self.Q_star[obs, :]
        q_max = np.max(q_list)
        choice_list = np.where(q_list == q_max)[0]
        return np.random.choice(choice_list)

    def sample(self, obs):
        if np.random.rand() < self.e_greed:
            return np.random.choice(self.act_dim)
        else:
            return self.predict(obs)

    def learn(self, obs, act, reward, next_obs, done):
        predict = self.Q_star[obs, act]
        td_target = reward + self.gamma * np.max(self.Q_star[next_obs, :])
        td_error = predict - td_target
        self.Q_star[obs, act] -= self.lr * td_error
