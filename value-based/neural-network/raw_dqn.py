import numpy as np
import torch
import torch.nn as nn
from utils.flappybird_wrapper import FlappyBirdWrapper
from utils.utils import (
    create_summary_writer,
    evaluate,
    load_model_params,
    save_model_params,
    train,
)

LEARNING_RATE = 1e-4
GAMMA = 0.98
E_GREED_INIT = 0.9
E_GREED_DECAY = 0.99
E_GREED_MIN = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RawDQNAgent:
    def __init__(self, obs_shape, act_dim):
        self.act_dim = act_dim
        self.e_greed = E_GREED_INIT

        # input (, 80, 80)
        self.model = nn.Sequential(
            *[
                # (, 80, 80) => (32, 40, 40)
                nn.Conv2d(
                    in_channels=obs_shape[0], out_channels=32, kernel_size=7, padding=3
                ),
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
                # (25600,) => (128,)
                nn.Linear(25600, 128),
                nn.SELU(inplace=True),
                # (128,) => (act_dim,)
                nn.Linear(128, act_dim),
            ]
        )
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def get_qvalue(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32, device=device)
        obs = torch.unsqueeze(obs, 0)
        obs = self.model(obs)
        obs = torch.squeeze(obs, 0)
        return obs

    def update_egreed(self):
        self.e_greed = max(E_GREED_MIN, self.e_greed * E_GREED_DECAY)
        return self.e_greed

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
            td_target = (
                reward
                + (1 - np.int(done)) * GAMMA * self.get_qvalue(next_obs).max(-1)[0]
            )

        loss = nn.functional.mse_loss(predict, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, filename):
        save_model_params(self.model, filename)

    def load(self, filename):
        load_model_params(self.model, filename)


def main():
    name = "raw_dqn_flappybird"
    writer = create_summary_writer(name)

    num = 10000000
    train_env = FlappyBirdWrapper()
    eval_env = FlappyBirdWrapper(display_screen=True)
    agent = RawDQNAgent(train_env.observation_space.shape, train_env.action_space.n)

    for episode in range(1, num):
        train(writer, episode, train_env, agent)
        if episode % 10 == 0:
            evaluate(writer, episode, eval_env, agent)
            agent.save(name)


if __name__ == "__main__":
    main()
