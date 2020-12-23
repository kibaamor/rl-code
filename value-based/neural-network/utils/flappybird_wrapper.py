#!/usr/bin/python
# coding=utf-8
from typing import Tuple

import cv2
import numpy as np
from gym import Env, spaces
from ple import PLE
from ple.games import FlappyBird

STACK = 3
FRAME_SIZE = (80, 80)


def preprocess(obs):
    obs = cv2.resize(obs, FRAME_SIZE)
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    # _, obs = cv2.threshold(obs, 159, 255, cv2.THRESH_BINARY)
    obs = np.reshape(obs, (1,) + FRAME_SIZE)
    obs = obs.astype(np.float32)
    return obs


class FlappyBirdWrapper(Env):
    def __init__(self, **kwargs):
        self.game = FlappyBird()
        self.p = PLE(self.game, **kwargs)
        self.p.init()
        self.action_set = self.p.getActionSet()

        self.observation_space = spaces.Space((STACK,) + FRAME_SIZE)
        self.action_space = spaces.Discrete(2)

        empty_frame = np.zeros(FRAME_SIZE, dtype=np.float32)
        self.obs = np.stack((empty_frame,) * STACK, axis=0)

    def _get_obs(self) -> np.ndarray:
        obs = self.p.getScreenRGB()
        obs = preprocess(obs)
        self.obs = np.concatenate((self.obs[1:, :, :], obs), axis=0)
        return self.obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, None]:
        reward = self.p.act(self.action_set[action])
        obs = self._get_obs()
        done = self.p.game_over()
        return obs, reward, done, None

    def reset(self) -> np.ndarray:
        self.p.reset_game()
        return self._get_obs()
