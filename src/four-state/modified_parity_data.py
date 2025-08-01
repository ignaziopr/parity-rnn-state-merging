import torch
import random
import itertools
import numpy as np
from torch.utils.data import IterableDataset


def modified_parity_label(seq):
    """
    DFA over {0,1,2} with 4 states E,O,T0,T1 as described.
    Returns 0 or 1.
    """
    state = 'E'
    for x in seq:
        if state == 'E':
            state = 'E' if x == 0 else ('O' if x == 1 else 'T0')
        elif state == 'O':
            state = 'O' if x == 0 else ('E' if x == 1 else 'T1')
        elif state == 'T0':
            state = 'E' if x in (0, 1) else 'T1'
        else:  # T1
            state = 'O' if x in (0, 1) else 'T0'
    return 0 if state in ('E', 'T0') else 1


class ParityStreamTrain(IterableDataset):
    """
    Exhaustive generator for all sequences of lengths 1..L_train over {0,1,2},
    with first symbol in {0,1}.
    """

    def __init__(self, L_train: int):
        self.L_train = L_train

    def __iter__(self):
        for length in range(1, self.L_train+1):
            # first token 0 or 1
            for first in (0, 1):
                # remaining tokens 0,1,2
                for tail in itertools.product((0, 1, 2), repeat=length-1):
                    seq = torch.tensor((first, *tail), dtype=torch.long)
                    lbl = torch.tensor(modified_parity_label(seq.tolist()))
                    yield seq, lbl


class ParityStreamVal(IterableDataset):
    """
    Random sampler for num_val sequences of fixed length L_val.
    """

    def __init__(self, num_val: int, L_val: int, seed: int | None = None):
        self.num_val = num_val
        self.L_val = L_val
        if seed is not None:
            random.seed(seed)

    def __iter__(self):
        for _ in range(self.num_val):
            first = random.choice((0, 1))
            tail = [random.choice((0, 1, 2)) for _ in range(self.L_val-1)]
            seq = torch.tensor((first, *tail), dtype=torch.long)
            lbl = torch.tensor(modified_parity_label(seq.tolist()))
            yield seq, lbl


class BufferedShuffle(IterableDataset):
    def __init__(self, base_ds, buffer_size=10000):
        self.base = base_ds
        self.buf = buffer_size

    def __iter__(self):
        it = iter(self.base)
        buf = []
        for _ in range(self.buf):
            try:
                buf.append(next(it))
            except StopIteration:
                break
        random.shuffle(buf)
        for item in it:
            i = random.randrange(len(buf))
            yield buf[i]
            buf[i] = item
        for item in buf:
            yield item
