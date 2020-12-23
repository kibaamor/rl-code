import random
from collections import deque, namedtuple

import torch

Transition = namedtuple("Transition", ("obs", "act", "reward", "next_obs", "done"))


class ExperienceReplay:
    def __init__(self, memory_capacity: int, batch_size: int):
        self.memory = deque(maxlen=memory_capacity)
        self.batch_size = batch_size

    def add(self, trans: Transition) -> None:
        self.memory.append(trans)

    def can_sample(self) -> bool:
        return len(self.memory) >= self.batch_size

    def sample(self, device: torch.device) -> Transition:
        batch = random.sample(self.memory, self.batch_size)
        batch = Transition(*(zip(*batch)))
        obs = torch.FloatTensor(batch.obs).to(device)
        act = torch.LongTensor(batch.act).unsqueeze(1).to(device)
        reward = torch.FloatTensor(batch.reward).to(device)
        next_obs = torch.FloatTensor(batch.next_obs).to(device)
        done = torch.LongTensor(batch.done).to(device)
        return Transition(obs, act, reward, next_obs, done)
