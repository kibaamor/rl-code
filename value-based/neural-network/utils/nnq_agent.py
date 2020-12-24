import os
import pathlib
from abc import ABC, abstractmethod
from copy import deepcopy
from os import path
from typing import Optional

import numpy as np
import torch
from gym import Env
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter

PAREND_DIR = pathlib.Path(__file__).parent.parent.resolve()
SAVED_MODEL_DIR = path.join(PAREND_DIR, "saved_model")
SUMMARY_BASE_DIR = path.join(PAREND_DIR, "runs")


def save_ckpt(model: Module, filename: str) -> bool:
    """Save checkpoint to filename

    return True if success else False
    """

    if not path.isdir(SAVED_MODEL_DIR):
        os.mkdir(SAVED_MODEL_DIR)

    fullpath = path.join(SAVED_MODEL_DIR, filename)

    model = deepcopy(model)
    model.to(torch.device("cpu"))  # always save model on cpu

    torch.save(model.state_dict(), fullpath)
    print(f"save checkpoint to {fullpath} success")
    return True


def load_ckpt(model: Module, filename: str) -> bool:
    """Load checkpoint from filename

    return True if success else False
    """

    fullpath = path.join(SAVED_MODEL_DIR, filename)
    if not path.isfile(fullpath):
        return False

    model.load_state_dict(torch.load(fullpath))
    print(f"load checkpoint from {fullpath} success")
    return True


def create_summary_writer(name: str) -> SummaryWriter:
    """Create a tensorboard writer"""

    if not path.isdir(SUMMARY_BASE_DIR):
        os.mkdir(SUMMARY_BASE_DIR)

    log_dir = path.join(SUMMARY_BASE_DIR, name)
    return SummaryWriter(log_dir=log_dir)


class NNQAgent(ABC):
    """Neural network based Q Learning agent"""

    def __init__(
        self,
        writer_name: str,
        ckpt_filename: str = None,
        network_name="network",
        lr: float = 1e-4,
        gamma: float = 0.98,
        eps_train: float = 0.9,
        eps_test: float = 0.001,
        eps_min: float = 0.01,
        eps_decay_per_update: float = 0.99,
        device: torch.device = None,
    ):
        self.ckpt_filename = (
            ckpt_filename if ckpt_filename is not None else writer_name + ".pth"
        )
        self.network_name = network_name
        self.lr = lr
        self.gamma = gamma
        self.eps_train = eps_train
        self.eps_test = eps_test
        self.eps_min = eps_min
        self.eps_decay_per_update = eps_decay_per_update
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.eps = self.eps_train
        self.writer = create_summary_writer(writer_name)
        self._train_step_count = 0

    def update_eps(self, episode: int, total_episode: int) -> float:
        self.eps = (
            self.eps_train - (self.eps_train - self.eps_min) * episode / total_episode
        )
        self.eps *= self.eps_decay_per_update
        return self.eps

    def save(self) -> None:
        network = getattr(self, self.network_name)
        if self.ckpt_filename is not None:
            save_ckpt(network, self.ckpt_filename)

    def load(self) -> None:
        network = getattr(self, self.network_name)
        if self.ckpt_filename is not None:
            load_ckpt(network, self.ckpt_filename)

    def get_act(self, obs: np.ndarray, is_train: bool) -> int:
        eps = self.eps if is_train else self.eps_test

        obs = torch.from_numpy(obs).to(self.device)
        obs.unsqueeze_(0)
        qvalue = self._predict_qvalue(obs, eps)
        act = qvalue.argmax(-1)[0].item()
        return act

    def train(
        self,
        episode: int,
        total_episode: int,
        env: Env,
        max_step: int,
    ) -> None:
        eps = self.update_eps(episode, total_episode)

        total_reward = 0
        obs = env.reset()

        for step in range(1, max_step):
            act = self.get_act(obs, True)
            next_obs, reward, done, _ = env.step(act)

            loss = self.learn(episode, obs, act, reward, next_obs, done)
            if loss is not None:
                self.writer.add_scalar("train/loss", loss, self._train_step_count)
            self._train_step_count += 1

            total_reward += reward
            obs = next_obs
            if done:
                break

        self.writer.add_scalar("train/eps", eps, episode)
        self.writer.add_scalar("train/step", step, episode)
        self.writer.add_scalar("train/reward", total_reward, episode)

    def test(self, episode: int, env: Env, max_step: int) -> None:
        total_reward = 0
        obs = env.reset()

        for step in range(1, max_step):
            act = self.get_act(obs, False)
            obs, reward, done, _ = env.step(act)

            total_reward += reward
            if done:
                break

        self.writer.add_scalar("test/eps", self.eps_test, episode)
        self.writer.add_scalar("test/step", step, episode)
        self.writer.add_scalar("test/reward", total_reward, episode)

    def train_test(
        self,
        train_env: Env,
        test_env: Env,
        load_cpkt: bool,
        total_episode: int,
        max_step_per_episode: int,
        episode_per_test: int,
        episode_per_save: int,
    ):
        if load_ckpt:
            self.load()

        for episode in range(1, total_episode):
            print(f"episode: {episode}")
            self.train(episode, total_episode, train_env, max_step_per_episode)

            if episode % episode_per_test == 0:
                self.test(episode, test_env, max_step_per_episode)
            if episode % episode_per_save == 0:
                self.save()

    def _predict_qvalue(self, obs: torch.Tensor, eps: float) -> torch.Tensor:
        network = getattr(self, self.network_name)
        with torch.no_grad():
            qvalue = network(obs)

        if not np.isclose(eps, 0.0):
            for i in range(len(qvalue)):
                if np.random.rand() < eps:
                    torch.rand(qvalue[i].shape, out=qvalue[i])

        return qvalue

    @abstractmethod
    def learn(
        self,
        episode: int,
        obs: np.ndarray,
        act: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        pass
