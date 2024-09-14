import random
import numpy as np
import torch
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    # torch.cuda.manual_seed(seed)
