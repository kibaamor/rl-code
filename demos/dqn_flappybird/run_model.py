#!/usr/bin/python
# coding=utf-8
import time
from flappybird_wrapper import FlappyBirdWrapper
from dqn_flappybird import Agent, pt_file


def main():
    env = FlappyBirdWrapper(display_screen=True)
    agent = Agent()
    agent.load()

    total_reward = 0
    obs = env.reset()
    # time.sleep(10)
    for _ in range(10000):
        time.sleep(1.0/120)
        act = agent.predict(obs)
        obs, reward, done, _ = env.step(act)
        total_reward += reward
        if done:
            print(f'total_reward: {total_reward}')
            break


if __name__ == '__main__':
    main()
