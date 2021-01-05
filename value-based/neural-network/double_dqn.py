import pathlib
from copy import deepcopy
from os.path import join

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.buffer import PrioritizedReplayBuffer, ReplayBuffer
from utils.flappybird_wrapper import FlappyBirdWrapper, create_network
from utils.misc import Collector, Policy, Tester, get_arg_parser, train


class DoubleDQNPolicy(Policy):
    def __init__(
        self,
        lr: float,
        gamma: float,
        use_selu: bool,
        dense_size: int,
        target_update_freq: int,
        tau: float,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = create_network(self.device, use_selu, dense_size)
        self.target_network = create_network(self.device, use_selu, dense_size)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        self.gamma = gamma
        self.eps = 0.0
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.update_count = 0

    def forward(self, obss: np.ndarray) -> np.ndarray:
        obss = torch.from_numpy(obss).to(self.device)
        qvals = self.network(obss)
        if not np.isclose(self.eps, 0.0):
            for i in range(len(qvals)):
                if np.random.rand() < self.eps:
                    torch.rand(qvals[i].shape, device=self.device, out=qvals[i])
        acts = qvals.argmax(-1)
        return acts.cpu().numpy()

    def soft_update_target(self):
        for target_param, local_param in zip(
            self.target_network.parameters(), self.network.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def update(self, buffer: PrioritizedReplayBuffer):
        is_prb = isinstance(buffer, PrioritizedReplayBuffer)
        batch = buffer.sample()

        obss = torch.FloatTensor(batch.obss).to(self.device)
        acts = torch.LongTensor(batch.acts).to(self.device).unsqueeze(1)
        rews = torch.FloatTensor(batch.acts).to(self.device)
        dones = torch.LongTensor(batch.acts).to(self.device)
        next_obss = torch.FloatTensor(batch.next_obss).to(self.device)
        weights = torch.FloatTensor(batch.weights).to(self.device) if is_prb else 1.0

        qval_pred = self.network(obss).gather(1, acts).squeeze()

        with torch.no_grad():
            acts_star = self.network(next_obss).argmax(-1).unsqueeze(1)
            qval_max = self.target_network(next_obss).gather(1, acts_star).squeeze()
        qval_targ = rews + (1 - dones) * self.gamma * qval_max

        td_err = qval_pred - qval_targ

        loss = (td_err.pow(2) * weights).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        if self.update_count >= self.target_update_freq:
            self.update_count -= self.target_update_freq
            self.soft_update_target()

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

        # for name, param in self.network.named_parameters():
        #     info[f"dist/network/{name}"] = param
        # for name, param in self.target_network.named_parameters():
        #     info[f"dist/target_network/{name}"] = param

        return info


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


def main():
    args = get_args()

    if args.alpha > 0.0:
        buffer = PrioritizedReplayBuffer(
            args.buffer_size,
            args.batch_size,
            args.alpha,
            args.beta,
        )
    else:
        buffer = ReplayBuffer(args.buffer_size, args.batch_size)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_env = FlappyBirdWrapper(caption=args.name, seed=args.seed)
    test_env = FlappyBirdWrapper(caption=args.name, seed=args.seed)

    collector = Collector(train_env, buffer, args.max_step_per_episode)
    tester = Tester(test_env, args.test_episode_per_step, args.max_step_per_episode)

    policy = DoubleDQNPolicy(
        args.lr,
        args.gamma,
        args.use_selu,
        args.dense_size,
        args.target_update_freq,
        args.tau,
    )

    here = pathlib.Path(__file__).parent.resolve()
    logdir = join(here, args.name)
    writer = SummaryWriter(logdir)

    def precollect(
        policy: DoubleDQNPolicy, epoch: int, steps: int, updates: int
    ) -> None:
        if steps <= 1e6:
            policy.eps = args.eps_collect - steps / 1e6 * (
                args.eps_collect - args.eps_collect_min
            )
        else:
            policy.eps = args.eps_collect_min
        writer.add_scalar("0_train/eps", policy.eps, steps)

    def preupdate(
        policy: DoubleDQNPolicy, epoch: int, steps: int, updates: int
    ) -> None:
        policy.eps = 0.0

    def pretest(policy: DoubleDQNPolicy, epoch: int, steps: int, updates: int) -> None:
        policy.eps = args.eps_test

    def save(policy: DoubleDQNPolicy, epoch: int, best_rew: float, rew: float) -> bool:
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
