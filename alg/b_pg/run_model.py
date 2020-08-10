#!/usr/bin/python
# coding=utf-8
import time
from pg import Agent, env, preprocess


def main():
    agent = Agent()
    agent.load()

    total_reward = 0
    obs = env.reset()
    env.render()
    for _ in range(10000):
        time.sleep(1.0 / 120)
        obs = preprocess(obs)
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
