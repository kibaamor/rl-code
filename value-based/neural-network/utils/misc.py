import time
from typing import Callable, List, Optional

import gym
import numpy as np
import torch
import tqdm
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from utils.buffer import PrioritizedReplayBuffer, ReplayBuffer


def mlp(
    sizes: List[int],
    activation: Optional[Callable[[], nn.Module]] = None,
    output_activation: Optional[Callable[[], nn.Module]] = None,
) -> nn.Module:
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        act = activation if i < len(sizes) - 2 else output_activation
        if act is not None:
            layers += [act()]
    return nn.Sequential(*layers)


class DuelingNetwork(nn.Module):
    def __init__(self, v_net, a_net):
        super().__init__()
        self.v_net = v_net
        self.a_net = a_net

    def forward(self, obs):
        v = self.v_net(obs)
        a = self.a_net(obs)
        return v - a.mean() + a


class Policy(nn.Module):
    def __init__(
        self,
        network: nn.Module,
        optimizer: torch.optim.Optimizer,
        gamma: float,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__()
        self.network = network.to(device)
        self.optimizer = optimizer
        self.gamma = gamma
        self.eps = 0.0
        self.device = device

    def forward(self, obss: np.ndarray) -> np.ndarray:
        obss = torch.FloatTensor(obss).to(self.device)
        qvals = self.network(obss)
        if not np.isclose(self.eps, 0.0):
            for i in range(len(qvals)):
                if np.random.rand() < self.eps:
                    torch.rand(qvals[i].shape, device=self.device, out=qvals[i])
        acts = qvals.argmax(-1)
        return acts.cpu().numpy()

    def update(self, buffer: ReplayBuffer) -> dict:
        is_prb = isinstance(buffer, PrioritizedReplayBuffer)
        batch = buffer.sample()

        obss = torch.FloatTensor(batch.obss).to(self.device)
        acts = torch.LongTensor(batch.acts).to(self.device).unsqueeze(1)
        rews = torch.FloatTensor(batch.acts).to(self.device)
        dones = torch.LongTensor(batch.acts).to(self.device)
        next_obss = torch.FloatTensor(batch.next_obss).to(self.device)
        weights = torch.FloatTensor(batch.weights).to(self.device) if is_prb else 1.0

        qval_pred = self.network(obss).gather(1, acts).squeeze()
        qval_targ = self.compute_target_q(next_obss, rews, dones)

        td_err = qval_pred - qval_targ

        loss = (td_err.pow(2) * weights).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        info = {
            "loss": loss.item(),
            "err_mean": td_err.mean().item(),
            "err_std": td_err.std().item(),
            "err_min": td_err.min().item(),
            "err_max": td_err.max().item(),
        }

        if is_prb:
            err_data = td_err.cpu().data.numpy()
            buffer.update_weight(batch.indexes, err_data)

            info["weights_mean"] = weights.mean().item()
            info["weights_std"] = weights.std().item()
            info["weights_min"] = weights.min().item()
            info["weights_max"] = weights.max().item()

        return info

    def compute_target_q(
        self,
        next_obss: torch.FloatTensor,
        rews: torch.FloatTensor,
        dones: torch.LongTensor,
    ) -> torch.FloatTensor:
        raise NotImplementedError


class Collector:
    def __init__(self, env: gym.Env, buffer: ReplayBuffer, max_step_per_episode: int):
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
    def __init__(self, env: gym.Env, episodes: int, max_step_per_episode: int):
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
            info = collector.collect(policy, collect_per_step)

        steps += collect_per_step
        write_scalar(writer, "0_train", info, steps)
        t.set_postfix({"step_per_s": info["step_per_s"]})

    def do_update(epoch: int, t: tqdm.tqdm) -> None:
        nonlocal updates

        if preupdate_fn:
            preupdate_fn(policy, epoch, steps, updates)

        losses = []
        for _ in range(update_per_step):
            updates += 1
            info = policy.update(collector.buffer)
            write_scalar(writer, "0_train", info, updates)
            losses.append(info["loss"])
        t.set_postfix({"loss_mean": np.mean(losses)})

    if warmup_size > 0:
        print(f"warming up for {warmup_size} experiences ... ", end="")
        with torch.no_grad():
            collector.collect(policy, warmup_size)
        print("done")

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
            t.set_postfix({"rew": last_rew})
