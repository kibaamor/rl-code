import numpy as np
import torch
import torch.nn as nn


class RawDQNAgent:
    def __init__(self, act_dim, lr=1e-4, e_greed=0.1, gamma=0.95):
        self.act_dim = act_dim
        self.e_greed = e_greed
        self.gamma = gamma

        # input (1, 80, 80)
        self.net = nn.Sequential(
            *[
                # (1, 80, 80) => (32, 40, 40)
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, padding=3),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7, padding=3),
                nn.SELU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                # (32, 40, 40) => (64, 20, 20)
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                nn.SELU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                # (64, 20, 20) => 25600
                nn.Flatten(start_dim=0),
                #
                nn.Linear(25600, 128),
                nn.SELU(inplace=True),
                # (128+1)*act_dim => (act_dim)
                nn.Linear(128, act_dim),
            ]
        )
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def get_qvalue(self, obs):
        obs = torch.unsqueeze(obs, 0)
        obs = self.net(obs)
        obs = torch.squeeze(obs, 0)
        return obs

    def predict(self, obs):
        with torch.no_grad():
            q_list = self.get_qvalue(obs).detach().numpy()
        q_max = np.max(q_list)
        choice_list = np.where(q_list == q_max)[0]
        act = np.random.choice(choice_list)
        return act

    def sample(self, obs):
        if np.random.rand() < self.e_greed:
            return np.random.choice(self.act_dim)
        else:
            return self.predict(obs)

    def learn(self, obs, act, reward, next_obs, done):
        predict = self.get_qvalue(obs)[act]

        with torch.no_grad():
            td_target = reward + self.gamma * self.get_qvalue(next_obs).max(-1)[0]

        loss = nn.functional.mse_loss(predict, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
