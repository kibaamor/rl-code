import pathlib
from copy import deepcopy
from os.path import join

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from utils.buffer import ReplayBuffer
from utils.misc import Policy, mlp, train
from utils.utils import create_collector_tester, get_arg_parser, make_gym_env


class DoubleDQNPolicy(Policy):
    def __init__(
        self,
        network: nn.Module,
        optimizer: torch.optim.Optimizer,
        gamma: float,
        target_update_freq: int,
        tau: float,
    ):
        super().__init__(network, optimizer, gamma)

        self.target_network = deepcopy(network).eval()
        self.target_network.load_state_dict(self.network.state_dict())

        self.target_update_freq = target_update_freq
        self.tau = tau
        self.update_count = 0

    def compute_target_q(
        self,
        next_obss: torch.FloatTensor,
        rews: torch.FloatTensor,
        dones: torch.LongTensor,
    ) -> torch.FloatTensor:
        with torch.no_grad():
            acts_star = self.network(next_obss).argmax(-1).unsqueeze(1)
            qval_max = self.target_network(next_obss).gather(1, acts_star).squeeze()
        qval_targ = rews + (1 - dones) * self.gamma * qval_max
        return qval_targ

    def update(self, buffer: ReplayBuffer) -> dict:
        info = super().update(buffer)

        self.update_count += 1
        if self.update_count >= self.target_update_freq:
            self.update_count -= self.target_update_freq
            self.soft_update_target()

        return info

    def soft_update_target(self) -> None:
        for target_param, local_param in zip(
            self.target_network.parameters(), self.network.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )


def get_args():
    parser = get_arg_parser("double-dqn")
    parser.add_argument(
        "--target-update-freq",
        type=int,
        default=10,
        metavar="N",
        help="target network update frequency",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        metavar="TAU",
        help="target network soft update parameters",
    )
    args = parser.parse_args()
    return args


def create_policy(args) -> Policy:
    env = make_gym_env(args)
    print(f"observation_space: {env.observation_space}")
    print(f"action_space: {env.action_space}")
    obs_n = env.observation_space.shape[0]
    act_n = env.action_space.n

    network = mlp(
        [obs_n] + [args.hidden_size] * args.layer_num + [act_n],
        nn.SELU if args.use_selu else nn.ReLU,
    )

    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    policy = DoubleDQNPolicy(
        network, optimizer, args.gamma, args.target_update_freq, args.tau
    )
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
