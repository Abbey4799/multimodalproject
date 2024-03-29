import torch
import numpy as np
import random


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sample_negative(nums, exclude):
    sampled = random.sample(nums, 2)
    for item in sampled:
        if item != exclude:
            return item
