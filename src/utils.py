import torch
import random
import numpy as np
import logging

def setup_logger(name="mace", level=logging.INFO):
    """Sets up a standardized logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_default_device():
    """Returns the default compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # For newer Macs, could add mps here, but for standard we use cpu
    return torch.device("cpu")
