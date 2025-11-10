import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (GCNConv, GATConv, GINConv, global_mean_pool, 
                                global_max_pool, global_add_pool, MessagePassing)
from torch_geometric.data import Data, DataLoader
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, LayerNorm
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import depolarizing_error, NoiseModel
import pandas as pd
from typing import List, Tuple

torch.manual_seed(0)
np.random.seed(0)

# ──────────────────────────────────────────────────────────────
# Custom Loss Functions for Column Prediction
# ──────────────────────────────────────────────────────────────
class ColumnL2Loss(nn.Module):
    """L2 loss for column vector predictions."""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        # Direct L2 norm of difference
        return torch.sqrt(torch.sum((pred - target) ** 2, dim=1)).mean()

class StochasticColumnLoss(nn.Module):
    """Combined loss ensuring column sums to 1 and matches target."""
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha  # Weight for stochasticity constraint
    
    def forward(self, pred, target):
        # Main reconstruction loss
        recon_loss = F.mse_loss(pred, target)
        
        # Stochasticity loss (columns should sum to 1)
        sum_loss = torch.abs(pred.sum(dim=1) - 1.0).mean()
        
        # Non-negativity penalty (if predictions go negative)
        neg_penalty = torch.relu(-pred).mean()
        
        return recon_loss + self.alpha * sum_loss + 0.01 * neg_penalty

class KLDivergenceLoss(nn.Module):
    """KL divergence loss treating columns as probability distributions."""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        pred_safe = pred + eps
        target_safe = target + eps
        
        # Normalize to ensure they're proper distributions
        pred_norm = pred_safe / pred_safe.sum(dim=1, keepdim=True)
        target_norm = target_safe / target_safe.sum(dim=1, keepdim=True)

        # KL divergence
        return (target_norm * torch.log(target_norm / pred_norm)).sum(dim=1).mean()