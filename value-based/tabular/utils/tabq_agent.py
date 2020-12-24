import time
from abc import ABC, abstractmethod

import numpy as np
from gym import Env


class TabQAgent(ABC):
    """Tabular base Q Learning agent"""

    def __init__(
        self,
        obs_n: int,
        act_n: int,
        lr: float = 0.1,
        eps: float = 0.1,
        gamma: float = 0.9,
    ):
        self.Q = np.zeros((obs_n, act_n))
        self.act_n = act_n
        self.lr = lr
        self.eps = eps
        self.gamma = gamma

    def _predict(self, obs: int) -> int:
        qvalue_list = self.Q[obs, :]
        choice_list = np.where(qvalue_list == qvalue_list.max())[0]
        return np.random.choice(choice_list)

    def _sample(self, obs: int) -> int:
        if np.random.rand() < self.eps:
            return np.random.choice(self.act_n)
        else:
            return self._predict(obs)

    def __call__(self, obs: int, is_train: bool) -> int:
        return self._sample(obs) if is_train else self._predict(obs)

    def train_test(
        self,
        env: Env,
        total_episode: int,
        max_step_per_episode: int,
    ) -> None:
        for episode in range(1, total_episode):
            print(f"episode: {episode}")
            self.train(env, max_step_per_episode)

        self.test(env, max_step_per_episode)

    def test(self, env: Env, max_step: int) -> None:
        total_step = 0
        total_reward = 0

        obs = env.reset()
        env.render()

        while True:
            act = self(obs, False)
            obs, reward, done, _ = env.step(act)

            total_step += 1
            total_reward += reward

            env.render()
            time.sleep(1.0 / 60)

            if done:
                break

        print(f"test total_step: {total_step}, total_reward: {total_reward}")

    @abstractmethod
    def train(self, env: Env, max_step: int) -> None:
        pass

    @abstractmethod
    def learn(*args, **kwargs) -> None:
        pass
