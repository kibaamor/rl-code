#!/usr/bin/python
# coding=utf-8
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from tensorboardX import SummaryWriter
import gym

script_path = os.path.split(os.path.realpath(__file__))[0]

pt_file = os.path.join(script_path, 'pg.pt')
env = gym.make('Pong-v0')

OBS_N = 80 * 80  # env.observation_space.shape[0]
ACT_N = env.action_space.n
HIDDEN_N = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0
GAMMA = 0.99
EPISODES_NUM = 8000000


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(OBS_N, HIDDEN_N)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(HIDDEN_N, ACT_N)
        self.fc2.weight.data.normal_(0, 0.1)

    def forward(self, obs):
        obs = torch.tanh(self.fc1(obs))
        action_prob = F.softmax(self.fc2(obs), dim=1)
        return action_prob


class Agent:
    def __init__(self):
        self.model = Model()
        self.optimizer = optim.Adam(self.model.parameters(
        ), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def predict(self, obs):
        obs = torch.FloatTensor(obs).unsqueeze(0)
        prob = self.model(obs).squeeze(0)
        dist = Categorical(prob)
        action = dist.sample()
        return action.item()

    def learn(self, obs_list, act_list, reward_list):
        obs_list = torch.FloatTensor(obs_list)
        act_list = torch.LongTensor(act_list)
        reward_list = torch.FloatTensor(reward_list)

        prob_list = self.model(obs_list)
        # loss = (-1.0 * prob_list.gather(1, act_list.unsqueeze(1)).log() * reward_list).mean()
        # https://pytorch.apachecn.org/docs/1.0/distributions.html
        loss = (-1.0 * Categorical(prob_list).log_prob(act_list)
                * reward_list).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self):
        torch.save(self.model.state_dict(), pt_file)
        print(pt_file + ' saved.')

    def load(self):
        self.model.load_state_dict(torch.load(pt_file))
        print(pt_file + ' loaded.')


def preprocess(image):
    """ 预处理 210x160x3 uint8 frame into 6400 (80x80) 1维 float vector """
    image = image[35:195]  # 裁剪
    image = image[::2, ::2, 0]  # 下采样，缩放2倍
    image[image == 144] = 0  # 擦除背景 (background type 1)
    image[image == 109] = 0  # 擦除背景 (background type 2)
    image[image != 0] = 1  # 转为灰度图，除了黑色外其他都是白色
    return image.astype(np.float).ravel()


def calc_reward_to_go(reward_list):
    for i in range(len(reward_list) - 2, -1, -1):
        reward_list[i] += GAMMA * reward_list[i + 1]

    eps = np.finfo(np.float32).eps.item()
    reward_list = (reward_list - np.mean(reward_list)) / \
        (np.std(reward_list) + eps)

    return reward_list


def train(writer, agent, episode):
    obs_list, act_list, reward_list = [], [], []
    obs = env.reset()
    for t in range(10000):
        obs = preprocess(obs)
        obs_list.append(obs)

        act = agent.predict(obs)
        act_list.append(act)

        obs, reward, done, _ = env.step(act)
        reward_list.append(reward)

        if done:
            writer.add_scalar('train/finish_step', t + 1, global_step=episode)
            writer.add_scalar('train/total_reward',
                              np.sum(reward_list), global_step=episode)

            reward_list = calc_reward_to_go(reward_list)
            loss = agent.learn(obs_list, act_list, reward_list)
            writer.add_scalar('train/value_loss', loss, global_step=episode)
            break


def evaluate(writer, agent, episode, render=True):
    total_reward = 0
    obs = env.reset()
    for t in range(10000):
        obs = preprocess(obs)
        act = agent.predict(obs)
        obs, reward, done, _ = env.step(act)
        total_reward += reward

        if render:
            env.render()
        if done:
            writer.add_scalar('evaluate/finish_step',
                              t + 1, global_step=episode)
            writer.add_scalar('evaluate/total_reward',
                              total_reward, global_step=episode)
            break


def main():
    agent = Agent()
    if os.path.exists(pt_file):
        agent.load()

    writer = SummaryWriter(os.path.join(script_path, 'PG'))
    for episode in range(EPISODES_NUM):
        print(f'episode: {episode}')
        train(writer, agent, episode)
        if episode % 10 == 9:
            evaluate(writer, agent, episode, False)
        if episode % 50 == 49:
            agent.save()


if __name__ == '__main__':
    main()
