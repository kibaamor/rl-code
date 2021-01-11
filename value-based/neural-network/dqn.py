import pathlib
from copy import deepcopy
from os.path import join

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from utils.misc import Policy, train
from utils.utils import create_collector_tester, create_network, get_arg_parser


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


def get_args():
    parser = get_arg_parser("dqn")
    args = parser.parse_args()
    return args


def create_policy(args) -> Policy:
    network = create_network(args)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    policy = DQNPolicy(network, optimizer, args.gamma)
    return policy


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    policy = create_policy(args)

    here = pathlib.Path(__file__).parent.resolve()
    logdir = join(here, args.name)
    writer = SummaryWriter(logdir)

    def precollect(policy: Policy, epoch: int, steps: int, updates: int) -> None:
        if steps <= 1e6:
            policy.eps = args.eps_collect - steps / 1e6 * (
                args.eps_collect - args.eps_collect_min
            )
        else:
            policy.eps = args.eps_collect_min
        writer.add_scalar("0_train/eps", policy.eps, steps)

    def preupdate(policy: Policy, epoch: int, steps: int, updates: int) -> None:
        policy.eps = 0.0

    def pretest(policy: Policy, epoch: int, steps: int, updates: int) -> None:
        policy.eps = args.eps_test

    def save(policy: Policy, epoch: int, best_rew: float, rew: float) -> bool:
        if rew <= best_rew:
            return True
        policy = deepcopy(policy).to(torch.device("cpu"))
        torch.save(policy.state_dict(), f"{logdir}/dqn_flappybird_{rew:.3f}.pth")
        return True

    collector, tester = create_collector_tester(args)

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
