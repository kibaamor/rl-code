import time

import gym
from qlearning import QLearningAgent
from sarsa import SarsaAgent
from utils.gridworld import CliffWalkingWapper


def train_qlearning(env, agent):
    agent.update_egreed()

    obs = env.reset()
    total_step = 0
    total_reward = 0

    while True:
        act = agent.sample(obs)
        next_obs, reward, done, _ = env.step(act)

        agent.learn(obs, act, reward, next_obs, done)

        obs = next_obs
        total_step += 1
        total_reward += reward

        if done:
            break

    return total_step, total_reward


def train_sarsa(env, agent):
    agent.update_egreed()

    total_step = 0
    total_reward = 0
    obs = env.reset()

    act = agent.sample(obs)

    while True:
        next_obs, reward, done, _ = env.step(act)
        next_act = agent.sample(next_obs)

        agent.learn(obs, act, reward, next_obs, next_act, done)

        obs = next_obs
        act = next_act

        total_step += 1
        total_reward += reward

        if done:
            break

    return total_step, total_reward


def evaluate(env, agent):
    total_step = 0
    total_reward = 0

    obs = env.reset()
    env.render()

    while True:
        act = agent.predict(obs)
        obs, reward, done, _ = env.step(act)

        total_step += 1
        total_reward += reward

        env.render()
        time.sleep(1.0 / 60)

        if done:
            break

    return total_step, total_reward


def train_evaluate(is_qlearning, num):
    env = gym.make("CliffWalking-v0")
    env = CliffWalkingWapper(env)
    obs_dim = env.observation_space.n
    act_dim = env.action_space.n

    agent = (
        QLearningAgent(obs_dim, act_dim)
        if is_qlearning
        else SarsaAgent(obs_dim, act_dim)
    )
    name = "Q-Learning" if is_qlearning else "Sarsa"

    print(f"training with {name}")
    for episode in range(num):
        if is_qlearning:
            total_step, total_reward = train_qlearning(env, agent)
        else:
            total_step, total_reward = train_sarsa(env, agent)

        if episode % 100 == 0:
            print(
                f"train episode: {episode}, total step: {total_step}, total reward: {total_reward}"
            )

    print(f"evaluate {name}")
    total_step, total_reward = evaluate(env, agent)
    print(f"evaluate result. total step: {total_step}, total reward: {total_reward}")

    env.close()


def main():
    num = 1000
    train_evaluate(False, num)
    train_evaluate(True, num)


if __name__ == "__main__":
    main()
