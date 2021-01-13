import argparse
from typing import Tuple

import gym
from torch import nn
from utils.buffer import PrioritizedReplayBuffer, ReplayBuffer
from utils.misc import Collector, DuelingNetwork, Tester, mlp


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
        "--game",
        type=str,
        default="MountainCar-v0",
        help="gym game name for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        metavar="M",
        help="learning rate step gamma",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,  # 0.6
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
        default=10000,
        metavar="N",
        help="replay buffer size",
    )
    parser.add_argument(
        "--warmup-size",
        type=int,
        default=256,
        metavar="N",
        help="warm up size for replay buffer(should greater than batch-size)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
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
        default=100,
        metavar="N",
        help="number of train step to epoch",
    )
    parser.add_argument(
        "--collect-per-step",
        type=int,
        default=8,
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
        "--test-episode-per-epoch",
        type=int,
        default=5,
        metavar="N",
        help="test episode per step",
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
        "--eps-collect-decay",
        type=float,
        default=0.9,
        metavar="DECAY",
        help="e-greeding for collecting experience",
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
        "--activation",
        type=str,
        default="relu",
        choices=["elu", "relu", "selu", "tanh", "ident"],
        help="activation function in network",
    )
    parser.add_argument(
        "--layer-num",
        type=int,
        default=2,
        metavar="N",
        help="hidden layer number",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=2048,
        metavar="N",
        help="hidden layer size",
    )
    parser.add_argument(
        "--dueling",
        action="store_true",
        default=False,
        help="use dueling network",
    )
    parser.add_argument(
        "--test-render-delay",
        type=float,
        default=0.0,
        metavar="F",
        help="render delay for test(0.0 to disable render)",
    )

    return parser


def make_gym_env(args) -> gym.Env:
    env = gym.make(args.game)
    env.seed(args.seed)
    return env


def create_network(args) -> nn.Module:
    env = make_gym_env(args)
    # print(f"observation_space: {env.observation_space}")
    # print(f"action_space: {env.action_space}")
    obs_n = env.observation_space.shape[0]
    act_n = env.action_space.n

    base_layers = [obs_n] + [args.hidden_size] * args.layer_num
    act_tab = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "tanh": nn.Tanh,
        "ident": None,
    }
    activation = act_tab[args.activation]

    if args.dueling:
        v_net = mlp(base_layers + [1], activation)
        a_net = mlp(base_layers + [act_n], activation)
        network = DuelingNetwork(v_net, a_net)
    else:
        network = mlp(base_layers + [act_n], activation)

    return network


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
    tester = Tester(
        test_env,
        args.test_episode_per_epoch,
        args.max_step_per_episode,
        args.test_render_delay,
    )
    return collector, tester
