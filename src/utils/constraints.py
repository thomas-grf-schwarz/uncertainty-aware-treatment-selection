import torch

def standardize(x):
    return (x - x.mean()) / x.std()

def soft_clamp(x, max):
    return max*torch.tanh(x)

def leaky_clamp(x, max, slope=0.01):
    return torch.where(x.abs() > max, slope*x, x)

def fix_sum(x, total):
    return total * x / x.sum()