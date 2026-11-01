"""Readouts operate on copies and cannot advance the main random streams."""
from contextlib import contextmanager
import random
import numpy as np
import torch

@contextmanager
def isolated_rng():
    py = random.getstate()
    np_state = np.random.get_state()
    cpu = torch.random.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.random.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
