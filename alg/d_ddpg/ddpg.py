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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

script_path = os.path.split(os.path.realpath(__file__))[0]

pt_file = os.path.join(script_path, 'ddpg.pt')
env = gym.make('Pendulum-v0')

OBS_N = env.observation_space.shape[0]  # (3,)
ACT_N = env.action_space.shape[0]  # (1,)
MAX_ACTION = float(env.action_space.high[0])  # array([2.], dtype=float32)

MODEL_INIT_WEIGHT = 3e-3
ACTOR_HIDDEN_N = 128
CRITIC_HIDDEN_N = 128
MEMORY_CAPACITY = 100000
BATCH_SIZE = 128
WARMUP_SIZE = BATCH_SIZE * 5
ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 1e-3
TAU = 1e-2
LEARN_FREQ = 8
GAMMA = 0.99
EXPLORATION_NOISE = 0.05
EPISODES_NUM = 8000000


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(OBS_N, ACTOR_HIDDEN_N)
        self.fc2 = nn.Linear(ACTOR_HIDDEN_N, ACTOR_HIDDEN_N)
        self.fc3 = nn.Linear(ACTOR_HIDDEN_N, ACT_N)

    def forward(self, obs):
        obs = F.relu(self.fc1(obs))
        obs = F.relu(self.fc2(obs))
        return MAX_ACTION * torch.tanh(self.fc3(obs))


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(OBS_N + ACT_N, CRITIC_HIDDEN_N)
        self.fc2 = nn.Linear(CRITIC_HIDDEN_N, CRITIC_HIDDEN_N)
        self.fc3 = nn.Linear(CRITIC_HIDDEN_N, 1)

    def forward(self, obs, act):
        concat = torch.cat([obs, act], dim=1)
        concat = F.relu(self.fc1(concat))
        concat = F.relu(self.fc2(concat))
        return self.fc3(concat)


class Agent:
    def __init__(self):
        self.actor = Actor().to(device)
        self.target_actor = Actor().to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_opt = optim.Adam(
            self.actor.parameters(), lr=ACTOR_LEARNING_RATE)

        self.critic = Critic().to(device)
        self.target_critic = Critic().to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_opt = optim.Adam(
            self.critic.parameters(), lr=CRITIC_LEARNING_RATE)

        self.memory = deque(maxlen=MEMORY_CAPACITY)
        self.update_count = 0

    def predict(self, obs):
        obs = torch.FloatTensor(obs).to(device)
        return self.actor(obs).cpu().detach().numpy()

    def store_transition(self, trans):
        self.memory.append(trans)

    def learn(self, writer):
        assert WARMUP_SIZE >= BATCH_SIZE
        if len(self.memory) < WARMUP_SIZE:
            return

        self.update_count += 1

        batch = random.sample(self.memory, BATCH_SIZE)
        batch = Transition(*(zip(*batch)))
        s0 = torch.FloatTensor(batch.state).to(device)
        a0 = torch.FloatTensor(batch.action).to(device)
        r1 = torch.FloatTensor(batch.reward).to(device).unsqueeze(1)
        s1 = torch.FloatTensor(batch.next_state).to(device)
        d1 = torch.LongTensor(batch.done).to(device).unsqueeze(1)

        with torch.no_grad():
            q_target = r1 + GAMMA * \
                (1 - d1) * self.target_critic(s1, self.target_actor(s1))
        q_current = self.critic(s0, a0)
        critic_loss = F.mse_loss(q_current, q_target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        writer.add_scalar('train/critic_loss',
                          critic_loss.item(), self.update_count)

        actor_loss = -self.critic(s0, self.actor(s0)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        writer.add_scalar('train/actor_loss',
                          actor_loss.item(), self.update_count)

        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(
                TAU * param.data + (1 - TAU) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(
                TAU * param.data + (1 - TAU) * target_param.data)

    def save(self):
        torch.save(self.actor.state_dict(), pt_file + '.actor')
        torch.save(self.critic.state_dict(), pt_file + '.critic')
        print(pt_file + ' saved.')

    def load(self):
        self.actor.load_state_dict(torch.load(pt_file + '.actor'))
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.critic.load_state_dict(torch.load(pt_file + '.critic'))
        self.target_critic.load_state_dict(self.critic.state_dict())
        print(pt_file + ' loaded.')


def train(writer, agent, episode):
    obs = env.reset()
    total_reward = 0
    for t in range(10000):
        act = agent.predict(obs)
        act = np.clip(np.random.normal(act, EXPLORATION_NOISE),
                      env.action_space.low, env.action_space.high)
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

    if os.path.exists(pt_file + '.actor') and os.path.exists(pt_file + '.critic'):
        agent.load()

    writer = SummaryWriter(os.path.join(script_path, 'DDPG'))
    for episode in range(EPISODES_NUM):
        print(f'episode: {episode}')
        train(writer, agent, episode)
        if episode % 10 == 9:
            evaluate(writer, agent, episode)
        if episode % 50 == 49:
            agent.save()


if __name__ == '__main__':
    main()
