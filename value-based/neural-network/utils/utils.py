import pathlib
from copy import deepcopy
from os import path

import torch
from torch.utils.tensorboard import SummaryWriter

PAREND_DIR = pathlib.Path(__file__).parent.parent.resolve()
SAVED_MODEL_DIR = path.join(PAREND_DIR, "saved_model")
SUMMARY_BASE_DIR = path.join(PAREND_DIR, "runs")


def save_model_params(model, filename):
    """Save model parameters to filename

    return True if success else False
    """

    filename += ".pth"
    fullpath = path.join(SAVED_MODEL_DIR, filename)

    net = deepcopy(model)
    net.to(torch.device("cpu"))  # always save model on cpu

    torch.save(net.state_dict(), fullpath)
    print(f"save params to {fullpath} success")
    return True


def load_model_params(model, filename):
    """Load model parammeters from filename

    return True if success else False
    """

    filename += ".pth"
    fullpath = path.join(SAVED_MODEL_DIR, filename)
    if not path.isfile(fullpath):
        return False

    model.load_state_dict(torch.load(fullpath))
    print(f"load params from {fullpath} success")
    return True


def create_summary_writer(name):
    """Create a tensorboard writer"""

    log_dir = path.join(SUMMARY_BASE_DIR, name)
    return SummaryWriter(log_dir=log_dir, comment=name)


def train(writer, episode, env, agent, max_step=10000):
    """Train agent with env"""

    e_greed = agent.update_egreed()

    total_reward = 0

    obs = env.reset()

    for t in range(max_step):
        act = agent.sample(obs)
        next_obs, reward, done, _ = env.step(act)

        agent.learn(obs, act, reward, next_obs, done)

        obs = next_obs
        total_reward += reward

        if done:
            break

    writer.add_scalar("train/e_greed", e_greed, global_step=episode)
    writer.add_scalar("train/total_step", t + 1, global_step=episode)
    writer.add_scalar("train/total_reward", total_reward, global_step=episode)


def evaluate(writer, episode, env, agent, max_step=10000):
    """evaluate agent with env"""

    total_reward = 0

    obs = env.reset()

    for t in range(10000):
        with torch.no_grad():
            act = agent.predict(obs)

        obs, reward, done, _ = env.step(act)

        total_reward += reward

        if done:
            break

    writer.add_scalar("evaluate/total_step", t + 1, global_step=episode)
    writer.add_scalar("evaluate/total_reward", total_reward, global_step=episode)
