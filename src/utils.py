"""
utils.py

This module contains helper functions.
"""

import random
import numpy as np
import torch


def set_random_seed(seed: int = 42):
    """
    Set the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
