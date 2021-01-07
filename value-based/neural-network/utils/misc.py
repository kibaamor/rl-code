import argparse
import time
from typing import Callable, List, Optional

import numpy as np
import torch
import tqdm
from gym import Env
from torch import nn
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter
from utils.buffer import ReplayBuffer


def mlp(
    sizes: List[int],
    activation: Callable[[], nn.Module] = nn.ReLU,
    output_activation: Callable[[], nn.Module] = nn.Identity,
) -> nn.Module:
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


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
        if k.startswith("dist/"):
            writer.add_histogram(k[5:], v, steps)
        else:
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

        with tqdm.tqdm(total=step_per_epoch, desc=f"Epoch #{epoch}", ascii=True) as t:
            while t.n < t.total:
                do_collect(epoch, t)
                do_update(epoch, t)
                t.update(1)

        policy.eval()
        last_rew = do_test(epoch)


def get_arg_parser(desc: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "name",
        type=str,
        help="name for this train",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        metavar="LR",
        help="learning rate",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.9,
        metavar="M",
        help="learning rate step gamma",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        metavar="ALPHA",
        help="alpha parameter for prioritized replay buffer(0 to use replay buffer)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.4,
        metavar="BETA",
        help="beta parameter for prioritized replay buffer",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=20000,
        metavar="N",
        help="replay buffer size",
    )
    parser.add_argument(
        "--warmup-size",
        type=int,
        default=256 * 4,
        metavar="N",
        help="warm up size for replay buffer(should greater than batch-size)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        metavar="N",
        help=(
            "batch size for training(should greater than collect-per-step"
            + " when using replay buffer)"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10000,
        metavar="N",
        help="number of epochs to train",
    )
    parser.add_argument(
        "--step-per-epoch",
        type=int,
        default=1000,
        metavar="N",
        help="number of train step to epoch",
    )
    parser.add_argument(
        "--collect-per-step",
        type=int,
        default=64,
        metavar="N",
        help="number of experience to collect per train step",
    )
    parser.add_argument(
        "--update-per-step",
        type=int,
        default=1,
        metavar="N",
        help="number of policy updating per train step",
    )
    parser.add_argument(
        "--max-step-per-episode",
        type=int,
        default=1000,
        metavar="N",
        help="max step per game episode",
    )
    parser.add_argument(
        "--test-episode-per-step",
        type=int,
        default=5,
        metavar="N",
        help="test episode per step",
    )
    parser.add_argument(
        "--eps-collect",
        type=float,
        default=0.1,
        metavar="EPS",
        help="e-greeding for collecting experience",
    )
    parser.add_argument(
        "--eps-collect-min",
        type=float,
        default=0.01,
        metavar="EPS",
        help="minimum e-greeding for collecting experience",
    )
    parser.add_argument(
        "--eps-test",
        type=float,
        default=0.01,
        metavar="EPS",
        help="e-greeding for testing policy",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="S",
        help="random seed",
    )

    parser.add_argument(
        "--use-selu",
        action="store_true",
        help="use selu or relu function in network",
    )
    parser.add_argument(
        "--layer-num",
        type=int,
        default=1,
        metavar="N",
        help="hidden layer number",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        metavar="N",
        help="hidden layer size",
    )

    return parser
