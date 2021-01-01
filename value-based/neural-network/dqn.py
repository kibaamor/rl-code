import argparse
import pathlib
from copy import deepcopy
from os.path import join

import numpy as np
import torch
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter
from utils.buffer import PrioritizedReplayBuffer
from utils.flappybird_wrapper import FlappyBirdWrapper, create_network
from utils.misc import Collector, Policy, Tester, train


class DQNPolicy(Policy):
    def __init__(self, lr: float, gamma: float, use_relu: bool, dense_size: int):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = create_network(self.device, use_relu, dense_size)
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


def get_args():
    parser = argparse.ArgumentParser(
        description="DQN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "name",
        type=str,
        help="name for this train",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=20000,
        metavar="N",
        help="prioritized replay buffer size",
    )
    parser.add_argument(
        "--warmup-size",
        type=int,
        default=1000,
        metavar="N",
        help="warm up size for prioritized replay buffer",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="batch size for training",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.9,
        metavar="ALPHA",
        help="alpha parameter for prioritized replay buffer",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        metavar="BETA",
        help="beta parameter for prioritized replay buffer",
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
        default=3,
        metavar="N",
        help="test episode per step",
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
        "--epochs",
        type=int,
        default=1000000,
        metavar="N",
        help="number of epochs to train",
    )
    parser.add_argument(
        "--step-per-epoch",
        type=int,
        default=10,
        metavar="N",
        help="number of train step to epoch",
    )
    parser.add_argument(
        "--collect-per-step",
        type=int,
        default=128,
        metavar="N",
        help="number of experience to collect per train step",
    )
    parser.add_argument(
        "--update-per-step",
        type=int,
        default=128,
        metavar="N",
        help="number of policy updating per train step",
    )

    parser.add_argument(
        "--eps-collect",
        type=float,
        default=0.9,
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
        default=0.9,
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
        "--use-relu",
        action="store_true",
        help="use relu or selu function in network",
    )
    parser.add_argument(
        "--dense-size",
        type=int,
        default=256,
        metavar="D",
        help="dense size in network",
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    buffer = PrioritizedReplayBuffer(
        args.buffer_size,
        args.batch_size,
        args.alpha,
        args.beta,
    )

    train_env = FlappyBirdWrapper(caption=args.name, seed=args.seed)
    test_env = FlappyBirdWrapper(caption=args.name, seed=args.seed)

    collector = Collector(train_env, buffer, args.max_step_per_episode)
    tester = Tester(test_env, args.test_episode_per_step, args.max_step_per_episode)

    policy = DQNPolicy(args.lr, args.gamma, args.use_relu, args.dense_size)

    here = pathlib.Path(__file__).parent.resolve()
    logdir = join(here, args.name)
    writer = SummaryWriter(logdir)

    def precollect(policy: DQNPolicy, epoch: int, steps: int, updates: int) -> None:
        policy.eps = (
            args.eps_collect
            - (args.eps_collect - args.eps_collect_min) * epoch / args.epochs
        )
        writer.add_scalar("0_train/eps", policy.eps, steps)

    def preupdate(policy: DQNPolicy, epoch: int, steps: int, updates: int) -> None:
        policy.eps = 0.0

    def pretest(policy: DQNPolicy, epoch: int, steps: int, updates: int) -> None:
        policy.eps = args.eps_test

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
        args.warmup_size,
        args.epochs,
        args.step_per_epoch,
        args.collect_per_step,
        args.update_per_step,
        args.batch_size,
        precollect_fn=precollect,
        preupdate_fn=preupdate,
        pretest_fn=pretest,
        save_fn=save,
    )


if __name__ == "__main__":
    main()
