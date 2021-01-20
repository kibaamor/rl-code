import pathlib
from argparse import ArgumentParser
from copy import deepcopy
from os.path import join
from pprint import pprint
from typing import Callable, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from utils.misc import Policy, train, watch
from utils.utils import create_collector, create_network, create_tester, get_arg_parser


class DQNPolicy(Policy):
    def __init__(
        self,
        network: nn.Module,
        optimizer: torch.optim.Optimizer,
        gamma: float,
    ):
        super().__init__(network, optimizer, gamma)

    def compute_target_q(
        self,
        next_obss: torch.FloatTensor,
        rews: torch.FloatTensor,
        dones: torch.LongTensor,
    ) -> torch.FloatTensor:
        with torch.no_grad():
            qval_max = self.network(next_obss).max(-1)[0]
        qval_targ = rews + (1 - dones) * self.gamma * qval_max
        return qval_targ


def get_args(parser_hook: Optional[Callable[[ArgumentParser], None]] = None):
    parser = get_arg_parser("dqn")
    if parser_hook is not None:
        parser_hook(parser)
    args = parser.parse_args()
    return args


def create_policy(args) -> Policy:
    network = create_network(args)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    policy = DQNPolicy(network, optimizer, args.gamma)

    if args.ckpt is not None:
        policy.load_state_dict(torch.load(args.ckpt))
        print(f"load checkpoint policy from file '{args.ckpt}'")

    return policy


def train_dqn(args) -> None:
    pprint(args.__dict__)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    policy = create_policy(args)

    here = pathlib.Path(__file__).parent.resolve()
    logdir = join(here, args.name)
    writer = SummaryWriter(logdir)

    def precollect(epoch: int, steps: int, updates: int) -> None:
        eps = args.eps_collect * (args.eps_collect_gamma ** epoch)
        policy.eps = eps if eps > args.eps_collect_min else args.eps_collect_min
        writer.add_scalar("train/eps", policy.eps, steps)

    def preupdate(epoch: int, steps: int, updates: int) -> None:
        policy.eps = 0.0

    def pretest(epoch: int, steps: int, updates: int) -> None:
        policy.eps = args.eps_test

    def save(epoch: int, best_rew: float, rew: float) -> bool:
        nonlocal policy

        if rew <= best_rew:
            return True

        policy = deepcopy(policy).to(torch.device("cpu"))
        torch.save(policy.state_dict(), f"{logdir}/dqn_{args.game}_{rew:.2f}.pth")
        if args.max_reward is not None:
            return rew < args.max_reward
        return True

    collector = create_collector(args)
    tester = create_tester(args)

    best_rew = train(
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
        max_loss=args.max_loss,
        precollect_fn=precollect,
        preupdate_fn=preupdate,
        pretest_fn=pretest,
        save_fn=save,
    )
    print(f"best rewards: {best_rew}")


def watch_dqn(args) -> None:
    policy = create_policy(args)
    tester = create_tester(args)
    watch(policy, tester, args.epochs)


if __name__ == "__main__":
    args = get_args()
    if args.watch:
        watch_dqn(args)
    else:
        train_dqn(args)
