import argparse
from typing import Tuple

import gym
from utils.buffer import PrioritizedReplayBuffer, ReplayBuffer
from utils.misc import Collector, Tester


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


def make_gym_env(args) -> gym.Env:
    env = gym.make("CartPole-v1")
    env.seed(args.seed)
    return env


def create_collector_tester(args) -> Tuple[Collector, Tester]:
    if args.alpha > 0.0:
        buffer = PrioritizedReplayBuffer(
            args.buffer_size, args.batch_size, args.alpha, args.beta
        )
    else:
        buffer = ReplayBuffer(args.buffer_size, args.batch_size)

    train_env = make_gym_env(args)
    test_env = make_gym_env(args)
    collector = Collector(train_env, buffer, args.max_step_per_episode)
    tester = Tester(test_env, args.test_episode_per_step, args.max_step_per_episode)
    return collector, tester
