#!/usr/bin/python
# coding=utf-8
import numpy as np
import gym


class Agent:
    def __init__(self, obs_n, act_n, lr=0.1, e_greed=0.1, gamma=0.9):
        self.Q = np.zeros((obs_n, act_n))
        self.act_n = act_n
        self.lr = lr
        self.e_greed = e_greed
        self.gamma = gamma

    def predict(self, obs):
        q_list = self.Q[obs, :]
        q_max = np.max(q_list)
        choice_list = np.where(q_list == q_max)[0]
        return np.random.choice(choice_list)

    def sample(self, obs):
        if np.random.rand() < self.e_greed:
            return np.random.choice(self.act_n)
        else:
            return self.predict(obs)

    def learn_by_sarsa(self, obs, act, reward, next_obs, next_act, done):
        predict_q = self.Q[obs, act]
        target_q = reward
        if not done:
            target_q += self.gamma * self.Q[next_obs, next_act]
        self.Q[obs, act] += self.lr * (target_q - predict_q)

    def learn_by_qlearning(self, obs, act, reward, next_obs, done):
        predict_q = self.Q[obs, act]
        target_q = reward
        if not done:
            target_q += self.gamma * np.max(self.Q[next_obs, :])
        self.Q[obs, act] += self.lr * (target_q - predict_q)


def train(env, agent, sarsa):
    total_step = 0
    total_reward = 0

    obs = env.reset()
    act = agent.sample(obs)
    while True:
        next_obs, reward, done, _ = env.step(act)

        if sarsa:
            next_act = agent.sample(next_obs)
            agent.learn_by_sarsa(obs, act, reward, next_obs, next_act, done)
            act = next_act
        else:
            agent.learn_by_qlearning(obs, act, reward, next_obs, done)
            act = agent.sample(next_obs)

        obs = next_obs

        total_step += 1
        total_reward += reward

        if done:
            break

    print(f"train total step: {total_step}, total reward: {total_reward}")


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

        if done:
            break

    print(f"evaluate total step: {total_step}, total reward: {total_reward}")


def main():
    env = gym.make("CliffWalking-v0")
    obs_n = env.observation_space.n
    act_n = env.action_space.n

    agent = Agent(obs_n, act_n)
    sarsa = False

    for episode in range(1000):
        print(f"episode: {episode}")
        train(env, agent, sarsa)

    evaluate(env, agent)


if __name__ == '__main__':
    main()
