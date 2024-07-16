import torch
import numpy as np
import torch.nn as nn
from src.utils.rnn import permute_rnn_style


def rbf_kernel(X, sigma=1.0):
    """
    Computes the RBF (Gaussian) kernel between rows of X.
    :param X: A 2D tensor of shape (n, d)
    :param sigma: The bandwidth parameter for the RBF kernel
    :return: A 2D tensor of shape (n, n) representing the kernel matrix
    """
    pairwise_dists = torch.cdist(X, X, p=2) ** 2
    K = torch.exp(-pairwise_dists / (2 * sigma ** 2))
    return K


def center_kernel(K):
    """
    Centers the kernel matrix K.
    :param K: A 2D tensor of shape (n, n) representing the kernel matrix
    :return: A 2D tensor of shape (n, n) representing the centered kernel matrix
    """
    n = K.size(0)
    unit = torch.ones((n, n), device=K.device) / n
    K_centered = K - unit @ K - K @ unit + unit @ K @ unit
    return K_centered


class HSICLoss(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma
    
    def forward_step(self, X, Y):
        """
        Computes the HSIC loss between X and Y at each time step.
        :param X: A 3D tensor of shape (N, D1, L)
        :param Y: A 3D tensor of shape (N, D2, L)
        :return: A scalar tensor representing the HSIC loss
        """

        # Compute the RBF kernel matrices
        Kx = rbf_kernel(X, self.sigma)
        Ky = rbf_kernel(Y, self.sigma)
        
        # Center the kernel matrices
        Kx_centered = center_kernel(Kx)
        Ky_centered = center_kernel(Ky)
        
        # Compute the HSIC value
        return torch.trace(Kx_centered @ Ky_centered) / (X.size(0) - 1) ** 2
    
    @permute_rnn_style
    def forward(self, X, Y):
        hsic = 0.0
        for l in range(X.shape[-1]):
            hsic += self.forward_step(X[..., l], Y[..., l])   
        return hsic


def get_past_targets(batch):
    return batch['outcome_history'][:, 1:, :], \
        batch['treatment_history'][:, 1:, :]


def get_future_targets(batch):
    return batch['outcomes'], batch['treatments']


def get_future_outcome_targets(batch):
    return batch['outcomes']


def get_past_outcome_targets(batch):
    return batch['outcome_history'][:, 1:, :]


def root_mean_square_error(input, target):
    errors = (input - target)**2
    return np.sqrt(errors.mean())