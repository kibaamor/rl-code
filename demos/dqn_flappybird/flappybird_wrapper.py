#!/usr/bin/python
# coding=utf-8
import numpy as np
from gym import spaces
from ple import PLE
from ple.games import FlappyBird


class FlappyBirdWrapper:
    def __init__(self, **kwargs):
        self.game = FlappyBird()
        self.p = PLE(self.game, **kwargs)
        self.action_set = self.p.getActionSet()

        self.observation_space = spaces.Discrete(3)
        self.action_space = spaces.Discrete(2)

    def _get_obs(self):
        state = self.game.getGameState()
        dist_to_pipe_horz = state["next_pipe_dist_to_player"]
        dist_to_pipe_bottom = state["player_y"] - state["next_pipe_top_y"]
        velocity = state['player_vel']
        return np.array([dist_to_pipe_horz, dist_to_pipe_bottom, velocity])

    def reset(self):
        self.p.reset_game()
        return self._get_obs()

    def step(self, action):
        reward = self.p.act(self.action_set[action])
        obs = self._get_obs()
        done = self.p.game_over()
        return obs, reward, done, None
