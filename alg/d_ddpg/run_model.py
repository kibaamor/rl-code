#!/usr/bin/python
# coding=utf-8
from ddpg import Agent, env


def main():
    agent = Agent()
    agent.load()

    total_reward = 0
    obs = env.reset()
    env.render()
    for _ in range(10000):
        act = agent.predict(obs)
        obs, reward, done, _ = env.step(act)
        total_reward += reward
        env.render()
        if done:
            print(f'total_reward: {total_reward}')
            env.close()
            break


if __name__ == '__main__':
    main()
