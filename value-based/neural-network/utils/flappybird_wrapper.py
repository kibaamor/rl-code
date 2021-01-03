#!/usr/bin/python
# coding=utf-8
import os
from typing import Tuple

import cv2
import numpy as np
import pygame
import torch
from gym import Env, spaces
from PIL import Image
from ple import PLE
from ple.games import FlappyBird
from torch import nn


class FlappyBirdWrapper(Env):
    def __init__(
        self,
        caption="Flappy Bird",
        stack_num: int = 4,
        frame_skip: int = 4,
        frame_size: Tuple[int, int] = (84, 84),
        display_screen: bool = False,
        force_fps: bool = True,
        seed: int = 24,
    ):
        self.game = FlappyBird()
        self.p = PLE(
            self.game,
            display_screen=display_screen,
            force_fps=force_fps,
            frame_skip=frame_skip,
            rng=seed,
        )
        self.p.init()
        self.action_set = self.p.getActionSet()
        pygame.display.set_caption(caption)

        self.stack_num = stack_num
        self.frame_size = frame_size

        self.observation_space = spaces.Space((stack_num,) + frame_size)
        self.action_space = spaces.Discrete(2)

        empty_frame = np.zeros(frame_size, dtype=np.float32)
        self.obs = np.stack((empty_frame,) * stack_num, axis=0)

    def preprocess(self, obs: np.ndarray) -> np.ndarray:
        obs = cv2.resize(obs, self.frame_size)
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # _, obs = cv2.threshold(obs, 159, 255, cv2.THRESH_BINARY)
        obs = np.reshape(obs, (1,) + self.frame_size)
        obs = obs.astype(np.float32) / 255
        return obs

    def _get_obs(self) -> np.ndarray:
        obs = self.p.getScreenRGB()
        obs = self.preprocess(obs)
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

    def save_screen(self, filename: str, preprocessed: bool = True) -> None:
        obs = self.p.getScreenRGB()
        if preprocessed:
            obs = self.preprocess(obs)[0]
            obs = obs * 255
            obs = obs.astype(np.uint8)
            obs = np.transpose(obs, axes=(1, 0))
        else:
            obs = np.transpose(obs, axes=(1, 0, 2))

        mode = "L" if preprocessed else "RGB"
        img = Image.fromarray(obs, mode)
        img.save(filename)


def create_network(
    device: torch.device,
    use_selu: bool,
    dense_size: int,
) -> nn.Module:
    def lu():
        return nn.SELU(inplace=True) if use_selu else nn.ReLU(inplace=True)

    # input (4, 84, 84)
    network = nn.Sequential(
        *[
            # (4, 84, 84) => (32, 20, 20)
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            lu(),
            # (32, 20, 20) => (64, 9, 9)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            lu(),
            # (64, 9, 9) => (64, 7, 7)
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            lu(),
            # (64, 7, 7) => 7*7*64=3136
            nn.Flatten(),
            nn.Linear(3136, dense_size),
            lu(),
            # => (2,)
            nn.Linear(dense_size, 2),
        ]
    )
    network.to(device)
    return network


def main():
    import pathlib

    here = pathlib.Path(__file__).parent.resolve()
    saved_screen = os.path.join(here, "saved_screen")
    if not os.path.isdir(saved_screen):
        os.mkdir(saved_screen)

    env = FlappyBirdWrapper()
    env.reset()

    # skip some frame
    for _ in range(20):
        _, reward, _, _ = env.step(np.random.randint(env.action_space.n))
        print(111, reward)

    env.save_screen(f"{saved_screen}/flappybird.png", False)
    env.save_screen(f"{saved_screen}/preprocessed_flappybird.png", True)


if __name__ == "__main__":
    main()
