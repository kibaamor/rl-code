#!/usr/bin/python
# coding=utf-8
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque
from tensorboardX import SummaryWriter
import gym

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

script_path = os.path.split(os.path.realpath(__file__))[0]

# Use Double-Q Learning ?
USE_DBQN = False

pt_file = os.path.join(script_path, 'dbqn.pt' if USE_DBQN else 'dqn.pt')
env = gym.make('MountainCar-v0')

OBS_N = env.observation_space.shape[0]
ACT_N = env.action_space.n
HIDDEN_N = 2048
MEMORY_CAPACITY = 10000
WARMUP_SIZE = 256
BATCH_SIZE = 128
MODEL_SYNC_COUNT = 20
LEARNING_RATE = 1e-3
LEARN_FREQ = 8
WEIGHT_DECAY = 0
GAMMA = 0.99
E_GREED = 0.1
E_GREED_DEC = 1e-6
E_GREED_MIN = 0.01
EPISODES_NUM = 8000000


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(OBS_N, HIDDEN_N)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc1.bias.data.zero_()
        self.fc2 = nn.Linear(HIDDEN_N, HIDDEN_N)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc2.bias.data.zero_()
        self.fc3 = nn.Linear(HIDDEN_N, HIDDEN_N)
        self.fc3.weight.data.normal_(0, 0.1)
        self.fc3.bias.data.zero_()
        self.fc4 = nn.Linear(HIDDEN_N, ACT_N)
        self.fc4.weight.data.normal_(0, 0.1)
        self.fc4.bias.data.zero_()

    def forward(self, obs):
        obs = F.relu(self.fc1(obs))
        obs = F.relu(self.fc2(obs))
        obs = F.relu(self.fc3(obs))
        action_val = self.fc4(obs)
        return action_val


class Agent:
    def __init__(self):
        self.model = Model()
        self.target_model = Model()
        self.target_model.load_state_dict(self.model.state_dict())

        self.memory = deque(maxlen=MEMORY_CAPACITY)
        self.optimizer = optim.Adam(self.model.parameters(
        ), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.loss_func = nn.MSELoss()

        self.e_greed = E_GREED
        self.update_count = 0

    def update_egreed(self):
        self.e_greed = max(E_GREED_MIN, self.e_greed - E_GREED_DEC)

    def predict(self, obs):
        q_val = self.model(torch.FloatTensor(obs)).detach().numpy()
        q_max = np.max(q_val)
        choice_list = np.where(q_val == q_max)[0]
        return np.random.choice(choice_list)

    def sample(self, obs):
        if np.random.rand() < self.e_greed:
            return np.random.choice(ACT_N)
        return self.predict(obs)

    def store_transition(self, trans):
        self.memory.append(trans)

    def learn(self, writer):
        assert WARMUP_SIZE >= BATCH_SIZE
        if len(self.memory) < WARMUP_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        batch = Transition(*(zip(*batch)))
        s0 = torch.FloatTensor(batch.state)
        a0 = torch.LongTensor(batch.action).unsqueeze(1)
        r1 = torch.FloatTensor(batch.reward)
        s1 = torch.FloatTensor(batch.next_state)
        d1 = torch.LongTensor(batch.done)

        q_pred = self.model(s0).gather(1, a0).squeeze()
        with torch.no_grad():
            if USE_DBQN:
                acts = self.model(s1).max(1)[1].unsqueeze(1)
                q_target = self.target_model(s1).gather(1, acts).squeeze(1)
            else:
                q_target = self.target_model(s1).max(1)[0]

            q_target = r1 + GAMMA * (1 - d1) * q_target
        loss = self.loss_func(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        writer.add_scalar('train/value_loss', loss.item(), self.update_count)
        self.update_count += 1
        if self.update_count % MODEL_SYNC_COUNT == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def save(self):
        torch.save(self.model.state_dict(), pt_file)
        print(pt_file + ' saved.')

    def load(self):
        self.model.load_state_dict(torch.load(pt_file))
        self.target_model.load_state_dict(self.model.state_dict())
        print(pt_file + ' loaded.')


def train(writer, agent, episode):
    agent.update_egreed()
    obs = env.reset()
    total_reward = 0
    for t in range(10000):
        act = agent.sample(obs)
        next_obs, reward, done, _ = env.step(act)

        trans = Transition(obs, act, reward, next_obs, done)
        agent.store_transition(trans)

        if t % LEARN_FREQ == 0:
            agent.learn(writer)

        obs = next_obs
        total_reward += reward
        if done or t >= 9999:
            writer.add_scalar('train/finish_step', t + 1, global_step=episode)
            writer.add_scalar('train/total_reward',
                              total_reward, global_step=episode)
            break


def evaluate(writer, agent, episode):
    total_reward = 0
    obs = env.reset()
    for t in range(10000):
        act = agent.predict(obs)
        obs, reward, done, _ = env.step(act)
        total_reward += reward

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

    writer = SummaryWriter(os.path.join(
        script_path, 'DBQN' if USE_DBQN else 'DQN'))
    for episode in range(EPISODES_NUM):
        print(f'episode: {episode}')
        train(writer, agent, episode)
        if episode % 10 == 9:
            evaluate(writer, agent, episode)
        if episode % 50 == 49:
            agent.save()


if __name__ == '__main__':
    main()
