#!/usr/bin/python3
# coding=utf-8
import gym
import numpy as np
from gym import Env
from utils.gridworld import CliffWalkingWapper
from utils.policy import Policy


class SarsaPolicy(Policy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, env: Env, max_step: int) -> None:
        total_step = 0
        total_reward = 0
        obs = env.reset()
        act = self(obs, True)

        while True:
            next_obs, reward, done, _ = env.step(act)
            next_act = self(next_obs, True)

            self.learn(obs, act, reward, next_obs, next_act, done)

            obs = next_obs
            act = next_act
            total_step += 1
            total_reward += reward
            if done:
                break

        print(f"train total_step: {total_step}, total_reward: {total_reward}")

    def learn(
        self,
        obs: int,
        act: int,
        reward: float,
        next_obs: int,
        next_act: int,
        done: bool,
    ) -> None:
        qvalue_predict = self.Q[obs, act]

        qvalue_target = (
            reward + (1 - np.int(done)) * self.gamma * self.Q[next_obs, next_act]
        )
        td_error = qvalue_predict - qvalue_target
        self.Q[obs, act] -= self.lr * td_error


def main():
    env = gym.make("CliffWalking-v0")
    env = CliffWalkingWapper(env)
    obs_n = env.observation_space.n
    act_n = env.action_space.n
    agent = SarsaPolicy(obs_n, act_n)

    total_episode = 1000
    max_step_per_episode = 100
    agent.train_test(env, total_episode, max_step_per_episode)


if __name__ == "__main__":
    main()
