import random
from collections import deque, namedtuple

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ExperienceReplay:
    def __init__(self, memory_capacity, batch_size):
        self.memory = deque(maxlen=memory_capacity)
        self.batch_size = batch_size

    def add(self, trans):
        self.memory.append(trans)

    def sample(self):
        batch = random.sample(self.memory, self.batch_size)
        batch = Transition(*(zip(*batch)))
        return batch
