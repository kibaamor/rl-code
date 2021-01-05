from typing import Any, Optional, Union

import numpy as np


class Batch:
    def __init__(
        self,
        obss: np.ndarray,
        acts: np.ndarray,
        rews: np.ndarray,
        dones: np.ndarray,
        next_obss: np.ndarray,
        indexes: np.ndarray = None,
        weights: Optional[np.ndarray] = None,
    ):
        self.obss = obss
        self.acts = acts
        self.rews = rews
        self.dones = dones
        self.next_obss = next_obss
        self.indexes = indexes
        self.weights = weights


class ReplayBuffer:
    """Experience replay

    >>> buffer = ReplayBuffer(10, 2)
    >>> for i in range(3):
    ...     assert i == len(buffer)
    ...     buffer.add(1, 2, 3, False, 4)
    >>> for i in range(20):
    ...     assert 2 == len(buffer)
    >>> batch = buffer.sample()
    >>> len(batch.obss) == len(batch.acts) == len(batch.rews) == len(batch.dones)
    True
    >>> len(batch.dones) == len(batch.next_obss) == len(batch.indexes)
    True
    """

    def __init__(self, buffer_size: int, batch_size: int):
        assert buffer_size >= batch_size

        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.obss: np.ndarray = None
        self.acts: np.ndarray = None
        self.rews: np.ndarray = None
        self.dones: np.ndarray = None
        self.next_obss: np.ndarray = None
        self.indexes = np.arange(buffer_size)
        self.index = 0
        self.fulled = False

    def __len__(self) -> int:
        return self.batch_size if self.fulled else self.index

    def _add(self, name: str, value: Any) -> None:
        value = np.array(value)
        batch = getattr(self, name, None)
        if batch is None:
            batch = np.empty((self.batch_size,) + value.shape)
            setattr(self, name, batch)
        batch[self.index] = value

    def add(self, obs: Any, act: Any, rew: float, done: bool, next_obs: Any) -> None:
        self._add("obss", obs)
        self._add("acts", act)
        self._add("rews", rew)
        self._add("dones", done)
        self._add("next_obss", next_obs)

        self.index += 1
        if self.index >= self.batch_size:
            self.fulled = True
            self.index -= self.batch_size

    def sample(self) -> Batch:
        assert len(self) >= self.batch_size

        indexes = np.random.choice(self.indexes[: len(self)], self.batch_size, False)
        return Batch(
            obss=self.obss[indexes],
            acts=self.acts[indexes],
            rews=self.rews[indexes],
            dones=self.dones[indexes],
            next_obss=self.next_obss[indexes],
            indexes=indexes,
        )


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized replay buffer

    >>> buffer = PrioritizedReplayBuffer(10, 2, 0.9, 1.0)
    >>> for i in range(3):
    ...     assert i == len(buffer)
    ...     buffer.add(1, 2, 3, False, 4)
    >>> for i in range(20):
    ...     assert 2 == len(buffer)
    >>> batch = buffer.sample()
    >>> len(batch.obss) == len(batch.acts) == len(batch.rews) == len(batch.dones)
    True
    >>> len(batch.dones) == len(batch.next_obss) == len(batch.indexes)
    True
    >>> buffer.update_weight(batch.indexes, batch.weights*0.5)
    """

    def __init__(self, buffer_size: int, batch_size: int, alpha: float, beta: float):
        assert alpha > 0.0 and beta >= 0.0

        super().__init__(buffer_size, batch_size)
        self.alpha = alpha
        self.beta = beta
        self.eps = np.finfo(np.float).eps.item()

        self.weights = SumTree(buffer_size)
        self.min_prio = 1.0
        self.max_prio = 1.0

    def add(self, obs: Any, act: Any, rew: float, done: bool, next_obs: Any):
        self.weights[self.index] = self.max_prio
        super().add(obs, act, rew, done, next_obs)

    def sample(self) -> Batch:
        assert len(self) >= self.batch_size

        scalars = np.random.rand(self.batch_size) * self.weights.reduce()
        indexes = self.weights.get_prefix_sum_index(scalars)
        weights = (self.weights[indexes] / self.min_prio) ** (-self.beta)
        return Batch(
            obss=self.obss[indexes],
            acts=self.acts[indexes],
            rews=self.rews[indexes],
            dones=self.dones[indexes],
            next_obss=self.next_obss[indexes],
            indexes=indexes,
            weights=weights,
        )

    def update_weight(self, indexes: np.ndarray, weights: np.ndarray) -> None:
        assert isinstance(indexes, np.ndarray)
        assert isinstance(weights, np.ndarray)

        weights = np.abs(weights) + self.eps
        self.weights[indexes] = weights ** self.alpha
        self.min_prio = min(self.min_prio, weights.min())
        self.max_prio = max(self.max_prio, weights.max())


class SumTree:
    """Segment tree with sum action

    >>> st = SumTree(10)
    >>> data = [3.0, 9.0, 2.0, 0.0, 6.0, 7.0, 1.0, 8.0, 5.0, 4.0]
    >>> for i, v in enumerate(data):
    ...     st[i] = v
    >>> len(st)
    10
    >>> list(st)
    [3.0, 9.0, 2.0, 0.0, 6.0, 7.0, 1.0, 8.0, 5.0, 4.0]
    >>> st[np.arange(10)]
    array([3., 9., 2., 0., 6., 7., 1., 8., 5., 4.])
    >>> st.reduce() == sum(st)
    True
    >>> for left in range(len(st)):
    ...     for right in range(left+1, len(st)):
    ...         assert st.reduce(left, right) == sum(data[left:right])
    >>> def get_prefix_sum_index(st, value):
    ...     for i in range(len(st)):
    ...         if st[i] >= value:
    ...             return i
    ...         value -= st[i]
    ...     assert False
    >>> for v in range(int(sum(st))):
    ...     assert get_prefix_sum_index(st, v) == st.get_prefix_sum_index(v)
    """

    def __init__(self, size: int):
        bound = 1
        while bound < size:
            bound <<= 1

        self.size = size
        self.bound = bound
        self.value = np.zeros([bound * 2], dtype=float)

    def __len__(self) -> int:
        return self.size

    def __iter__(self):
        left, right = self.bound, self.bound + self.size
        return iter(self.value[left:right])

    def __getitem__(self, index: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        assert isinstance(index, (int, np.ndarray))
        assert np.all(0 <= index) and np.all(index < self.size)

        return self.value[self.bound + index]

    def __setitem__(
        self, index: Union[int, np.ndarray], value: Union[float, np.ndarray]
    ) -> None:
        assert np.all(0 <= index) and np.all(index < self.size)

        if not isinstance(index, np.ndarray):
            index = np.array([index])
            value = np.array([value])

        index = index + self.bound  # DO NOT MODIFY INPUT
        self.value[index] = value
        while index[0] > 1:
            self.value[index >> 1] = self.value[index] + self.value[index ^ 1]
            index >>= 1

    def reduce(self, left: int = 0, right: Optional[int] = None) -> float:
        """Return sum(self[left:right])"""

        if left == 0 and right is None:
            return self.value[1]

        if right is None:
            right = self.size
        if right < 0:
            right += self.size
        left, right = left + self.bound, right + self.bound

        result = 0.0
        while left < right:
            if left & 1:
                result += self.value[left]
                left += 1
            left >>= 1
            if right & 1:
                right -= 1
                result += self.value[right]
            right >>= 1

        return result

    def get_prefix_sum_index(
        self, value: Union[float, np.ndarray]
    ) -> Union[int, np.ndarray]:
        """Return index which sum(self[:index-1]) < value <= sum(self[:index])"""

        single = False
        if not isinstance(value, np.ndarray):
            single = True
            value = np.array([value], dtype=float)
        assert np.all(0.0 <= value) and np.all(value <= self.value[1])

        value = value.copy()  # DO NOT MODIFY INPUT
        index = np.ones_like(value, dtype=np.int)
        while index[0] < self.bound:
            index <<= 1
            left_value = self.value[index]
            direction = left_value < value
            value -= left_value * direction
            index += direction
        index -= self.bound

        return index.item() if single else index


if __name__ == "__main__":
    import doctest

    doctest.testmod()
