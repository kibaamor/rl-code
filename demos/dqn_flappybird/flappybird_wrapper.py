#!/usr/bin/python
# coding=utf-8
import numpy as np
from gym import Env, spaces
from ple import PLE
from ple.games import FlappyBird


class FlappyBirdWrapper(Env):
    # 如果想把画面渲染出来，就传参display_screen=True
    def __init__(self, **kwargs):
        self.game = FlappyBird()
        self.p = PLE(self.game, **kwargs)
        self.action_set = self.p.getActionSet()

        # 3个输入状态：见函数self._get_obs
        self.observation_space = spaces.Discrete(3)
        # 两个输出状态：跳或者不跳
        self.action_space = spaces.Discrete(2)

    def _get_obs(self):
        # 获取游戏的状态
        state = self.game.getGameState()
        # 小鸟与它前面一对水管中下面那根水管的水平距离
        dist_to_pipe_horz = state["next_pipe_dist_to_player"]
        # 小鸟与它前面一对水管中下面那根水管的顶端的垂直距离
        dist_to_pipe_bottom = state["player_y"] - state["next_pipe_top_y"]
        # 获取小鸟的水平速度
        velocity = state['player_vel']
        # 将这些信息封装成一个数据返回
        return np.array([dist_to_pipe_horz, dist_to_pipe_bottom, velocity])

    def reset(self):
        self.p.reset_game()
        return self._get_obs()

    def step(self, action):
        reward = self.p.act(self.action_set[action])
        obs = self._get_obs()
        done = self.p.game_over()
        return obs, reward, done, dict()

    def seed(self, *args, **kwargs):
        pass

    def render(self, *args, **kwargs):
        pass
