import torch
import torch.nn as nn


class LayerNorm1D(nn.Module):
    def __init__(self, normalized_shape, dim_to_normalize, eps=1e-8):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, *normalized_shape))
        self.beta = nn.Parameter(torch.zeros(1, *normalized_shape))
        self.dim_to_normalize = dim_to_normalize
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=self.dim_to_normalize, keepdims=True)
        std = x.std(dim=self.dim_to_normalize, keepdims=True)
        return self.gamma * (x - mean) / std + self.beta    


class FixedLayerNorm1D(nn.Module):
    def __init__(self, normalized_shape, dim_to_normalize, eps=1e-8):
        super().__init__()
        self.dim_to_normalize = dim_to_normalize
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=self.dim_to_normalize, keepdims=True)
        std = x.std(dim=self.dim_to_normalize, keepdims=True)
        return (x - mean) / std


class ReversibleNorm(nn.Module):
    def __init__(self, normalized_shape, dim_to_normalize, eps=1e-8):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.dim_to_normalize = dim_to_normalize
        self.gamma = nn.Parameter(torch.ones(1, *normalized_shape))
        self.beta = nn.Parameter(torch.zeros(1, *normalized_shape))
        self.eps = eps
        self.mean = 0
        self.std = 1

    def forward(self, x):
        self.mean = x.mean(dim=self.dim_to_normalize, keepdims=True)
        self.std = x.std(dim=self.dim_to_normalize, keepdims=True) + self.eps
        return self.gamma * (x - self.mean) / self.std + self.beta
        
    def inverse(self, x):
        mean = x.mean(dim=self.dim_to_normalize, keepdims=True)
        std = x.std(dim=self.dim_to_normalize, keepdims=True) + self.eps
        return self.std * (x - mean) / std + self.mean
    

class TransferableNorm(nn.Module):

    def __init__(self, normalized_shape, dim_to_normalize, eps=1e-8):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.dim_to_normalize = dim_to_normalize
        self.gamma = nn.Parameter(torch.ones(1, *normalized_shape))
        self.beta = nn.Parameter(torch.zeros(1, *normalized_shape))
        self.eps = eps
        self.mean = 0
        self.std = 1
    
    def forward(self, target_seq, source_seq):
        self.mean = source_seq.mean(dim=self.dim_to_normalize, keepdims=True)
        self.std = source_seq.std(dim=self.dim_to_normalize, keepdims=True) + self.eps
        return self.gamma * ((target_seq - self.mean) / self.std) + self.beta
    
    def inverse(self, target_seq):
        mean = target_seq.mean(dim=self.dim_to_normalize, keepdims=True)
        std = target_seq.std(dim=self.dim_to_normalize, keepdims=True) + self.eps
        return self.std * ((target_seq - mean) / std) + self.mean


class LBackNorm(nn.Module):

    def __init__(self, normalized_shape, dim_to_normalize, l_back=8, eps=1e-8):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.dim_to_normalize = dim_to_normalize
        self.gamma = nn.Parameter(torch.ones(1, *normalized_shape))
        self.beta = nn.Parameter(torch.zeros(1, *normalized_shape))
        self.eps = eps
        self.mean = 0
        self.std = 1
        self.l_back = l_back
    
    def forward(self, target_seq, source_seq):
        L_future = target_seq.shape[-2]
        L_past = source_seq.shape[-2]
        L_past_effective = L_past + L_future - self.l_back

        nback_seq = torch.cat([
            source_seq[..., -L_past_effective:], target_seq],
            dim=-2,
        )

        self.mean = nback_seq.mean(dim=self.dim_to_normalize, keepdims=True)
        self.std = nback_seq.std(dim=self.dim_to_normalize, keepdims=True) + self.eps
        return self.gamma * ((target_seq - self.mean) / self.std) + self.beta
    
    def transfer(self, source_seq):
        return self.gamma * ((source_seq - self.mean) / self.std) + self.beta
    
    def inverse(self, target_seq):
        mean = target_seq.mean(dim=self.dim_to_normalize, keepdims=True)
        std = target_seq.std(dim=self.dim_to_normalize, keepdims=True) + self.eps
        return self.std * ((target_seq - mean) / std) + self.mean


class FixedAllBackNorm(nn.Module):

    def __init__(self, normalized_shape, dim_to_normalize, eps=1e-8):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.dim_to_normalize = dim_to_normalize
        self.eps = eps
        self.mean = 0
        self.std = 1

    def forward(self, target_seq, source_seq):

        all_seq = torch.cat([
            source_seq, target_seq],
            dim=-2,
        )

        self.mean = all_seq.mean(dim=self.dim_to_normalize, keepdims=True)
        self.std = all_seq.std(dim=self.dim_to_normalize, keepdims=True) + self.eps
        return (target_seq - self.mean) / self.std

    def transfer(self, source_seq):
        return (source_seq - self.mean) / self.std

    def inverse(self, target_seq):
        mean = target_seq.mean(dim=self.dim_to_normalize, keepdims=True)
        std = target_seq.std(dim=self.dim_to_normalize, keepdims=True) + self.eps
        return self.std * ((target_seq - mean) / (std + self.eps)) + self.mean


class FixedTransferableNorm(nn.Module):

    def __init__(self, normalized_shape, dim_to_normalize, eps=1e-8):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.dim_to_normalize = dim_to_normalize
        self.eps = eps
        self.mean = 0
        self.std = 1
    
    def forward(self, target_seq, source_seq):
        self.mean = source_seq.mean(dim=self.dim_to_normalize, keepdims=True)
        self.std = source_seq.std(dim=self.dim_to_normalize, keepdims=True) + self.eps
        return (target_seq - self.mean) / self.std

    def transfer(self, source_seq):
        return (source_seq - self.mean) / self.std

    def inverse(self, target_seq):
        mean = target_seq.mean(dim=self.dim_to_normalize, keepdims=True)
        std = target_seq.std(dim=self.dim_to_normalize, keepdims=True) + self.eps
        return self.std * ((target_seq - mean) / (std + self.eps)) + self.mean
