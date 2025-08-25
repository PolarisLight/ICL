import functools
from loguru import logger
import random
import numpy as np
import torch
import os

def core_module(cls):
    orig_init = cls.__init__

    @functools.wraps(cls.__init__)
    def new_init(self, *args):
        logger.bind(params=True).info(f"{'='*30}")
        logger.bind(params=True).info(f"Instantiating core module: {cls.__name__}")
        if hasattr(self, 'get_core_params'):
            core_params = self.get_core_params()
            logger.bind(params=True).info("Core hyperparameters and recommended tuning ranges:")
            for param, tuning_range in core_params.items():
                logger.bind(params=True).info(f"  {param} (Recommended range: {tuning_range})")
        logger.bind(params=True).info(f"{'='*30}")
        orig_init(self, *args)
    
    cls.__init__ = new_init
    return cls

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed_value)