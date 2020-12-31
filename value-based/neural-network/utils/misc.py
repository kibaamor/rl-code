import time
from typing import Callable, Optional

import numpy as np
import torch
import tqdm
from gym import Env
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter
from utils.buffer import ReplayBuffer


class Policy(Module):
    def __init__(self):
        super().__init__()

    def forward(self, obss: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def update(batch_size, buffer: ReplayBuffer) -> None:
        raise NotImplementedError


class Collector:
    def __init__(self, env: Env, buffer: ReplayBuffer, max_step_per_episode: int):
        self.buffer = buffer
        self.env = env
        self.max_step_per_episode = max_step_per_episode

        self.obs = self.env.reset()
        self.step = 0

    def collect(self, policy: Policy, steps: int):
        rews = []

        beg_t = time.time()
        for _ in range(steps):
            act = policy(np.array([self.obs]))[0]
            next_obs, rew, done, _ = self.env.step(act)

            self.buffer.add(self.obs, act, rew, done, next_obs)
            rews.append(rew)

            self.step += 1
            self.obs = next_obs
            if done or self.step >= self.max_step_per_episode:
                self.obs = self.env.reset()

        cost_t = time.time() - beg_t

        return {
            "rew_mean": np.mean(rews),
            "rew_std": np.std(rews),
            "rew_min": np.min(rews),
            "rew_max": np.max(rews),
            "step_per_s": steps / cost_t,
        }


class Tester:
    def __init__(self, env: Env, episodes: int, max_step_per_episode: int):
        self.env = env
        self.episodes = episodes
        self.max_step_per_episode = max_step_per_episode

    def test(self, policy: Policy):
        episode_rews = []
        episode_steps = []

        beg_t = time.time()
        for episode in range(1, 1 + self.episodes):
            rews = 0
            obs = self.env.reset()

            for step in range(1, 1 + self.max_step_per_episode):
                act = policy(np.array([obs]))[0]
                obs, rew, done, _ = self.env.step(act)
                rews += rew
                if done:
                    break

            episode_rews.append(rews)
            episode_steps.append(step)
        cost_t = time.time() - beg_t

        return {
            "rew_mean": np.mean(episode_rews),
            "rew_std": np.std(episode_rews),
            "rew_min": np.min(episode_rews),
            "rew_max": np.max(episode_rews),
            "step_mean": np.mean(episode_steps),
            "step_std": np.std(episode_steps),
            "step_min": np.min(episode_steps),
            "step_max": np.max(episode_steps),
            "step_per_s": np.sum(episode_steps) / cost_t,
            "ms_per_episode": 1000.0 * cost_t / self.episodes,
        }


def write_scalar(writer: SummaryWriter, prefix: str, info: dict, steps: int) -> None:
    for k, v in info.items():
        writer.add_scalar(f"{prefix}/{k}", v, steps)


def train(
    writer: SummaryWriter,
    policy: Policy,
    collector: Collector,
    tester: Tester,
    warmup_size: int,
    epochs: int,
    step_per_epoch: int,
    collect_per_step: int,
    update_per_step: int,
    batch_size: int,
    precollect_fn: Optional[Callable[[Policy, int], None]] = None,
    preupdate_fn: Optional[Callable[[Policy, int, int], None]] = None,
    pretest_fn: Optional[Callable[[Policy, int], None]] = None,
    save_fn: Optional[Callable[[Policy, int, float, float], bool]] = None,
) -> None:
    steps = 0
    updates = 0
    last_rew, best_rew = -np.inf, -np.inf

    def do_test(epoch: int) -> dict:
        if pretest_fn:
            pretest_fn(policy, epoch, steps, updates)

        with torch.no_grad():
            info = tester.test(policy)
        write_scalar(writer, "1_test", info, steps)
        return info["rew_mean"]

    def do_save(epoch: int) -> bool:
        nonlocal best_rew

        if save_fn:
            if not save_fn(policy, epoch, best_rew, last_rew):
                return False

        best_rew = max(best_rew, last_rew)
        return True

    def do_collect(epoch: int, t: tqdm.tqdm) -> None:
        nonlocal steps

        if precollect_fn:
            precollect_fn(policy, epoch, steps, updates)

        with torch.no_grad():
            if epoch == 1:
                collector.collect(policy, warmup_size)
            info = collector.collect(policy, collect_per_step)

        steps += collect_per_step
        write_scalar(writer, "0_train", info, steps)
        t.set_postfix({"step_per_s": info["step_per_s"]})

    def do_update(epoch: int, t: tqdm.tqdm) -> None:
        nonlocal updates

        if preupdate_fn:
            preupdate_fn(policy, epoch, steps, updates)

        for _ in range(update_per_step):
            updates += 1
            info = policy.update(collector.buffer)
            write_scalar(writer, "0_train", info, updates)
            t.set_postfix({"loss": info["loss"]})

    for epoch in range(1, 1 + epochs):
        policy.train()
        if not do_save(epoch):
            break

        with tqdm.tqdm(total=step_per_epoch, desc=f"Epoch #{epoch}") as t:
            while t.n < t.total:
                do_collect(epoch, t)
                do_update(epoch, t)
                t.update(1)

        policy.eval()
        last_rew = do_test(epoch)
