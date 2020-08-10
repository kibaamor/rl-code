#!/usr/bin/python
# coding=utf-8
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque
from tensorboardX import SummaryWriter
from flappybird_wrapper import FlappyBirdWrapper

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

script_path = os.path.split(os.path.realpath(__file__))[0]

pt_file = os.path.join(script_path, 'dqn_flappybird.pt')
env = FlappyBirdWrapper()

# 游戏状态的种类数
OBS_N = env.observation_space.n
# 游戏操作的种类数
ACT_N = env.action_space.n
# 隐藏层的大小
HIDDEN_N = 128
# 经验池的大小
MEMORY_CAPACITY = 10000
# 经验池预热的大小
WARMUP_SIZE = 256
# 每次学习时使用的经验数
BATCH_SIZE = 128
# target model同步的频率
MODEL_SYNC_COUNT = 20
# 学习率
LEARNING_RATE = 1e-3
# 学习的频率
LEARN_FREQ = 8
# 衰减因子
GAMMA = 0.99
# 探索的概率
E_GREED = 0.1
# 探索概率的减少量
E_GREED_DEC = 1e-6
# 探索概率的最小值
E_GREED_MIN = 0.01
# 训练的总回合数
EPISODES_NUM = 2000
# 是否使用Double-Q Learning
USE_DBQN = False


# 模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(OBS_N, HIDDEN_N)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc1.bias.data.zero_()
        self.fc2 = nn.Linear(HIDDEN_N, HIDDEN_N)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc2.bias.data.zero_()
        self.fc3 = nn.Linear(HIDDEN_N, ACT_N)
        self.fc3.weight.data.normal_(0, 0.1)
        self.fc3.bias.data.zero_()

    def forward(self, obs):
        obs = torch.tanh(self.fc1(obs))
        obs = torch.tanh(self.fc2(obs))
        action_val = self.fc3(obs)
        return action_val


# 智能体
class Agent:
    def __init__(self):
        # 模型
        self.model = Model().to(device)
        # 目标模型（Target Model）
        self.target_model = Model().to(device)
        # 同步目标模型
        self.target_model.load_state_dict(self.model.state_dict())

        # 经验池
        self.memory = deque(maxlen=MEMORY_CAPACITY)
        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        # Loss函数
        self.loss_func = nn.MSELoss()

        # 初始化探索概率
        self.e_greed = E_GREED
        # 更新次数的计数
        self.update_count = 0

    def update_egreed(self):
        # 更新探索概率
        self.e_greed = max(E_GREED_MIN, self.e_greed - E_GREED_DEC)

    def predict(self, obs):
        # 根据状态预测动作
        obs = torch.FloatTensor(obs).to(device)
        q_val = self.model(obs).cpu().detach().numpy()
        # 需要注意的是：如果多个动作的Q值都是最大时，要从这些中随机选一个
        q_max = np.max(q_val)
        choice_list = np.where(q_val == q_max)[0]
        return np.random.choice(choice_list)

    def sample(self, obs):
        # 采样动作
        if np.random.rand() < self.e_greed:
            # 探索
            return np.random.choice(ACT_N)
        # 利用已有经验
        return self.predict(obs)

    def store_transition(self, trans):
        # 保存经验
        self.memory.append(trans)

    def learn(self, writer):
        # 根据经验进行学习

        # 确保经验池里的经验以及足够多时，才进行学习
        assert WARMUP_SIZE >= BATCH_SIZE
        if len(self.memory) < WARMUP_SIZE:
            return

        # 从经验池中选取BATCH_SIZE条经验出来
        batch = random.sample(self.memory, BATCH_SIZE)
        batch = Transition(*(zip(*batch)))
        # 把这些经验都转换位Tensor
        s0 = torch.FloatTensor(batch.state).to(device)
        a0 = torch.LongTensor(batch.action).to(device).unsqueeze(1)
        r1 = torch.FloatTensor(batch.reward).to(device)
        s1 = torch.FloatTensor(batch.next_state).to(device)
        d1 = torch.LongTensor(batch.done).to(device)

        # DQN算法
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

        # 记录每次学习的loss
        writer.add_scalar('train/value_loss', loss.item(), self.update_count)
        self.update_count += 1
        # 每MODEL_SYNC_COUNT同步一下目标模型
        if self.update_count % MODEL_SYNC_COUNT == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def save(self):
        # 保存模型文件
        torch.save(self.model.state_dict(), pt_file)
        print(pt_file + ' saved.')

    def load(self):
        # 加载模型文件
        self.model.load_state_dict(torch.load(pt_file))
        # 目标模型要同步一下
        self.target_model.load_state_dict(self.model.state_dict())
        print(pt_file + ' loaded.')


def train(writer, agent, episode):
    # 训练模型

    # 更新探索率
    agent.update_egreed()
    # 重开一局
    obs = env.reset()
    # 这局获得的总奖励数
    total_reward = 0
    # 每局操作不超过10000次，防止死循环
    for t in range(10000):
        # 进行一个操作
        act = agent.sample(obs)
        next_obs, reward, done, _ = env.step(act)

        # 保存经验
        trans = Transition(obs, act, reward, next_obs, done)
        agent.store_transition(trans)

        # 每LEARN_FREQ次学习一次
        if t % LEARN_FREQ == 0:
            agent.learn(writer)

        obs = next_obs
        # 累加获得的奖励
        total_reward += reward
        if done or t >= 9999:
            # 记录这局游戏完成的步骤数
            writer.add_scalar('train/finish_step', t + 1, global_step=episode)
            # 记录这句游戏获得的总奖励数
            writer.add_scalar('train/total_reward',
                              total_reward, global_step=episode)
            break


def evaluate(writer, agent, episode):
    # 评估

    # 重开一局
    obs = env.reset()
    # 这局获得的总奖励数
    total_reward = 0
    # 每局操作不超过10000次，防止死循环
    for t in range(10000):
        # 预测一个操作
        act = agent.predict(obs)
        obs, reward, done, _ = env.step(act)

        # 累加获得的奖励
        total_reward += reward

        if done:
            # 记录这局游戏完成的步骤数
            writer.add_scalar('evaluate/finish_step',
                              t + 1, global_step=episode)
            # 记录这句游戏获得的总奖励数
            writer.add_scalar('evaluate/total_reward',
                              total_reward, global_step=episode)
            break


def main():
    # 创建智能体
    agent = Agent()

    # 加载训练好的模型文件
    if os.path.exists(pt_file):
        agent.load()

    # 创建tensorboard对象
    writer = SummaryWriter(os.path.join(script_path, 'DQN'))
    for episode in range(EPISODES_NUM):
        if episode % 10 == 0:
            print(f'episode: {episode}')

        # 训练
        train(writer, agent, episode)

        # 每训练10次就评估一次
        if episode % 10 == 9:
            evaluate(writer, agent, episode)

        # 每训练50次就保存模型文件
        if episode % 50 == 49:
            agent.save()


if __name__ == '__main__':
    main()
