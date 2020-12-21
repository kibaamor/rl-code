#!/usr/bin/python
# coding=utf-8
import cv2
import numpy as np
import torch
from gym import spaces
from PIL import Image
from ple import PLE
from ple.games import FlappyBird


def preprocess(obs):
    obs = cv2.resize(obs, (80, 80))
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    # _, obs = cv2.threshold(obs, 159, 255, cv2.THRESH_BINARY)
    obs = np.reshape(obs, (1, 80, 80))
    return obs


class FlappyBirdWrapper:
    def __init__(self, **kwargs):
        self.game = FlappyBird()
        self.p = PLE(self.game, force_fps=False, **kwargs)
        self.p.init()
        self.action_set = self.p.getActionSet()

        self.observation_space = spaces.Space((1, 80, 80))
        self.action_space = spaces.Discrete(2)
        self.num = 0

    def _get_obs(self):
        obs = self.p.getScreenRGB()
        obs = preprocess(obs)

        # if self.num < 1000:
        #     img = Image.fromarray(obs[0], "L")
        #     img.save(f"D:/{self.num}.png")
        #     self.num += 1

        obs = torch.FloatTensor(obs)
        return obs

    def reset(self):
        self.p.reset_game()
        return self._get_obs()

    def step(self, action):
        reward = self.p.act(self.action_set[action])
        obs = self._get_obs()
        done = self.p.game_over()
        return obs, reward, done, {}
