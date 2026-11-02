# imports.py
import os
import time
import math
import random
from argparse import Namespace

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, norm

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


def get_device(prefer_cuda: bool = True):
    """
    Returns the PyTorch compute device ('cuda' if an Nvidia GPU is available,
    otherwise 'cpu'), or None if PyTorch is not installed.
    """
    if not TORCH_AVAILABLE:
        return None
    if prefer_cuda and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


DEVICE = get_device(prefer_cuda=True)
