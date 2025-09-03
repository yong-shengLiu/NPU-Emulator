
from collections import deque
from typing import Optional, Tuple, List

class OperandQueue:
    """Simple FIFO queue used for staging vector operands per lane."""
    def __init__(self, depth: int = 4):
        self.q = deque(maxlen=depth)

    def push(self, vec: List[int]):
        if len(self.q) == self.q.maxlen:
            raise RuntimeError('OperandQueue full')
        self.q.append(list(vec))

    def pop(self) -> Optional[List[int]]:
        if not self.q:
            return None
        return self.q.popleft()

    def __len__(self):
        return len(self.q)
