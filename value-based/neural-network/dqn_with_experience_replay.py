from typing import Optional

import numpy as np
import torch
from torch.nn.functional import mse_loss
from utils.agent import Agent
from utils.experience_replay import ExperienceReplay, Transition
from utils.flappybird_wrapper import FlappyBirdWrapper, create_network


class DQNWithExperienceReplayAgent(Agent):
    def __init__(
        self,
        memory_capacity,
        batch_size,
        step_per_learn,
        writer_name: str,
        *args,
        **kwargs,
    ):
        super().__init__(writer_name, *args, **kwargs)
        self.exp_replay = ExperienceReplay(memory_capacity, batch_size)
        self.step_per_learn = step_per_learn

        self.learn_count = 0

        self.network = create_network(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

    def learn(
        self,
        episode: int,
        obs: np.ndarray,
        act: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        trans = Transition(obs, act, reward, next_obs, done)
        self.exp_replay.add(trans)

        self.learn_count += 1
        if self.learn_count >= self.step_per_learn:
            self.learn_count -= self.step_per_learn
            return self._do_learn(episode)

    def _do_learn(self, episode: int) -> Optional[float]:
        if not self.exp_replay.can_sample():
            return
        batch = self.exp_replay.sample(self.device)

        qvalue_predict = self.network(batch.obs).gather(1, batch.act).squeeze()

        qvalue_max = self._predict_qvalue(batch.next_obs, 0).max(-1)[0]
        qvalue_target = batch.reward + (1 - batch.done) * self.gamma * qvalue_max

        loss = mse_loss(qvalue_predict, qvalue_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


def main():
    name = "dqn_with_experience_replay"
    train_env = FlappyBirdWrapper(caption=name)
    test_env = FlappyBirdWrapper(caption=name, display_screen=True)
    memory_capacity = 100000
    batch_size = 128
    step_per_learn = 128
    agent = DQNWithExperienceReplayAgent(
        memory_capacity, batch_size, step_per_learn, name
    )

    agent.train_test(train_env, test_env)


if __name__ == "__main__":
    main()
