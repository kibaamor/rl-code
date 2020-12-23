from typing import Optional

import numpy as np
import torch
from gym.spaces import Discrete, Space
from torch import nn
from torch.nn.functional import mse_loss
from utils.agent import Agent
from utils.experience_replay import ExperienceReplay, Transition
from utils.flappybird_wrapper import FlappyBirdWrapper


class DQNWithExperienceReplayAgent(Agent):
    def __init__(
        self,
        obs_space: Space,
        act_space: Discrete,
        memory_capacity,
        batch_size,
        step_per_learn,
        writer_name: str,
        *args,
        **kwargs,
    ):
        super().__init__(writer_name, *args, **kwargs)
        self.exp_replay = ExperienceReplay(memory_capacity, batch_size)
        self.step_per_learn = step_per_learn

        self.learn_count = 0

        # input (, 80, 80)
        self.network = nn.Sequential(
            *[
                # (, 80, 80) => (32, 40, 40)
                nn.Conv2d(
                    in_channels=obs_space.shape[0],
                    out_channels=32,
                    kernel_size=7,
                    padding=3,
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
                nn.Flatten(),
                # (25600,) => (128,)
                nn.Linear(25600, 128),
                nn.SELU(inplace=True),
                # (128,) => (act_space.n,)
                nn.Linear(128, act_space.n),
            ]
        )
        self.network.to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

    def learn(
        self,
        episode: int,
        obs: np.ndarray,
        act: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        trans = Transition(obs, act, reward, next_obs, done)
        self.exp_replay.add(trans)

        self.learn_count += 1
        if self.learn_count >= self.step_per_learn:
            self.learn_count -= self.step_per_learn
            return self._do_learn(episode)

    def _do_learn(self, episode: int) -> Optional[float]:
        if not self.exp_replay.can_sample():
            return
        batch = self.exp_replay.sample(self.device)

        qvalue_predict = self.network(batch.obs).gather(1, batch.act).squeeze()

        qvalue_max = self._predict_qvalue(batch.next_obs, 0).max(-1)[0]
        qvalue_target = batch.reward + (1 - batch.done) * self.gamma * qvalue_max

        loss = mse_loss(qvalue_predict, qvalue_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


def main():
    train_env = FlappyBirdWrapper()
    test_env = FlappyBirdWrapper(display_screen=True)
    memory_capacity = 100000
    batch_size = 128
    step_per_learn = 128
    writer_name = "dqn_with_experience_replay"
    agent = DQNWithExperienceReplayAgent(
        train_env.observation_space,
        train_env.action_space,
        memory_capacity,
        batch_size,
        step_per_learn,
        writer_name,
    )

    agent.train_test(train_env, test_env)


if __name__ == "__main__":
    main()
