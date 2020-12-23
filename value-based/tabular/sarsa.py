#!/usr/bin/python3
# coding=utf-8
import numpy as np

LEARNING_RATE = 0.1
GAMMA = 0.98
E_GREED_INIT = 0.9
E_GREED_DECAY = 0.99
E_GREED_MIN = 0.01


class SarsaAgent:
    def __init__(self, obs_dim, act_dim):
        self.Q_pi = np.zeros((obs_dim, act_dim))
        self.act_dim = act_dim
        self.e_greed = E_GREED_INIT

    def update_egreed(self):
        self.e_greed = max(E_GREED_MIN, self.e_greed * E_GREED_DECAY)
        return self.e_greed

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
        td_target = reward + (1 - np.int(done)) * GAMMA * self.Q_pi[next_obs, next_act]
        td_error = predict - td_target
        self.Q_pi[obs, act] -= LEARNING_RATE * td_error
