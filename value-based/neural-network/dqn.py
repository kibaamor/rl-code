from copy import deepcopy

import numpy as np
import torch
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter
from utils.buffer import PrioritizedReplayBuffer
from utils.flappybird_wrapper import FlappyBirdWrapper, create_network
from utils.misc import Collector, Policy, Tester, train


class DQNPolicy(Policy):
    def __init__(self, lr: float, gamma: float):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = create_network(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.gamma = gamma
        self.eps = 0.0

    def forward(self, obss: np.ndarray) -> np.ndarray:
        obss = torch.from_numpy(obss).to(self.device)
        qvals = self.network(obss)
        if not np.isclose(self.eps, 0.0):
            for i in range(len(qvals)):
                if np.random.rand() < self.eps:
                    torch.rand(qvals[i].shape, device=self.device, out=qvals[i])
        acts = qvals.argmax(-1)
        return acts.cpu().numpy()

    def update(self, buffer: PrioritizedReplayBuffer):
        batch = buffer.sample()

        obss = torch.FloatTensor(batch.obss).to(self.device)
        acts = torch.LongTensor(batch.acts).to(self.device).unsqueeze(1)
        rews = torch.FloatTensor(batch.acts).to(self.device)
        dones = torch.LongTensor(batch.acts).to(self.device)
        next_obss = torch.FloatTensor(batch.next_obss).to(self.device)

        qval_pred = self.network(obss).gather(1, acts).squeeze()

        with torch.no_grad():
            qval_max = self.network(next_obss).max(-1)[0]
        qval_targ = rews + (1 - dones) * self.gamma * qval_max

        errors = torch.abs(qval_pred - qval_targ).cpu().data.numpy()
        buffer.update_weight(batch.indexes, errors)

        loss = mse_loss(qval_pred, qval_targ)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "err_mean": np.mean(errors),
            "err_std": np.std(errors),
            "err_min": np.min(errors),
            "err_max": np.max(errors),
        }


def main():
    buffer_size = 20000
    batch_size = 128
    buffer = PrioritizedReplayBuffer(buffer_size, batch_size, 0.9, 1.0)

    name = "dqn"
    train_env = FlappyBirdWrapper(caption=name)
    # test_env = FlappyBirdWrapper(caption=name, display_screen=True, force_fps=False)
    test_env = FlappyBirdWrapper(caption=name)

    max_step_per_episode = 1000
    test_per_step = 3
    collector = Collector(train_env, buffer, max_step_per_episode)
    tester = Tester(test_env, test_per_step, max_step_per_episode)

    warmup_size = 1000
    epochs = 100000
    step_per_epoch = 1
    collect_per_step = 128
    update_per_step = 1

    lr = 1e-4
    gamma = 0.98
    policy = DQNPolicy(lr, gamma)

    logdir = name
    writer = SummaryWriter(logdir)

    def precollect(policy: DQNPolicy, epoch: int, steps: int, updates: int) -> None:
        policy.eps = 0.9 - (0.9 - 0.01) * epoch / epochs
        writer.add_scalar("0_train/eps", policy.eps, steps)

    def preupdate(policy: DQNPolicy, epoch: int, steps: int, updates: int) -> None:
        policy.eps = 0.0

    def pretest(policy: DQNPolicy, epoch: int, steps: int, updates: int) -> None:
        policy.eps = 0.01

    def save(policy: DQNPolicy, epoch: int, best_rew: float, rew: float) -> bool:
        if rew <= best_rew:
            return True
        policy = deepcopy(policy).to(torch.device("cpu"))
        torch.save(policy.state_dict(), f"{logdir}/dqn_flappybird_{rew:.3f}.pth")
        return True

    train(
        writer,
        policy,
        collector,
        tester,
        warmup_size,
        epochs,
        step_per_epoch,
        collect_per_step,
        update_per_step,
        batch_size,
        precollect_fn=precollect,
        preupdate_fn=preupdate,
        pretest_fn=pretest,
        save_fn=save,
    )


if __name__ == "__main__":
    main()
