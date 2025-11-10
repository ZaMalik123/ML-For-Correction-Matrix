# 

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

# ──────────────────────────────────────────────────────────────
# GNN Models for Column Prediction
# ──────────────────────────────────────────────────────────────
class ColumnGNN(nn.Module):
    """GNN for predicting first column of assignment matrix."""
    def __init__(self, node_feat_dim, hidden_dim, output_dim, global_feat_dim=4,
                 n_layers=4, dropout=0.3, aggregation='multi', enforce_stochastic=True):
        super().__init__()
        
        self.aggregation = aggregation
        self.enforce_stochastic = enforce_stochastic
        
        # Node embedding
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Graph layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(n_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Global feature processing
        self.global_processor = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        # Determine final dimension
        if aggregation == 'multi':
            agg_dim = hidden_dim * 3
        else:
            agg_dim = hidden_dim
        final_dim = agg_dim + hidden_dim
        
        # Output decoder
        self.decoder = nn.Sequential(
            nn.Linear(final_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, batch, global_features=None):
        # Encode nodes
        x = self.node_encoder(x)
        
        # Message passing with residuals
        for conv, norm in zip(self.convs, self.norms):
            x_res = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = x + x_res
        
        # Pooling
        if self.aggregation == 'multi':
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x_sum = global_add_pool(x, batch)
            x = torch.cat([x_mean, x_max, x_sum], dim=1)
        else:
            x = global_mean_pool(x, batch)
        
        # Add global features
        if global_features is not None:
            global_encoded = self.global_processor(global_features)
            x = torch.cat([x, global_encoded], dim=1)
        
        # Decode
        out = self.decoder(x)
        
        # Enforce stochasticity if requested
        if self.enforce_stochastic:
            out = F.softmax(out, dim=1)
        else:
            out = torch.sigmoid(out)  # Keep values in [0,1]
        
        return out

class ColumnGAT(nn.Module):
    """Graph Attention Network for column prediction."""
    def __init__(self, node_feat_dim, hidden_dim, output_dim, global_feat_dim=4,
                 n_layers=3, heads=4, dropout=0.3, enforce_stochastic=True):
        super().__init__()
        
        self.enforce_stochastic = enforce_stochastic
        
        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        
        # First GAT layer
        self.convs.append(GATConv(hidden_dim, hidden_dim, heads=heads, concat=True))
        
        # Middle layers
        for _ in range(n_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True))
        
        # Last layer
        self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False))
        
        # Global processing
        self.global_processor = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Output
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, batch, global_features=None):
        x = self.node_encoder(x)
        
        # Attention layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = self.dropout(x)
        
        # Pool
        x = global_mean_pool(x, batch)
        
        # Add global features
        if global_features is not None:
            global_encoded = self.global_processor(global_features)
            x = torch.cat([x, global_encoded], dim=1)
        
        # Decode
        out = self.decoder(x)
        
        # Enforce stochasticity
        if self.enforce_stochastic:
            out = F.softmax(out, dim=1)
        else:
            out = torch.sigmoid(out)
        
        return out

class ColumnGIN(nn.Module):
    """Graph Isomorphism Network for column prediction."""
    def __init__(self, node_feat_dim, hidden_dim, output_dim, global_feat_dim=4,
                 n_layers=4, dropout=0.3, enforce_stochastic=True):
        super().__init__()
        
        self.n_layers = n_layers
        self.enforce_stochastic = enforce_stochastic
        
        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(n_layers):
            nn_model = Sequential(
                Linear(hidden_dim, hidden_dim),
                BatchNorm1d(hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(nn_model))
            self.batch_norms.append(BatchNorm1d(hidden_dim))
        
        # Global processing
        self.global_processor = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Jump knowledge
        self.jump = nn.Linear(n_layers * hidden_dim, hidden_dim)
        
        # Output
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, batch, global_features=None):
        x = self.node_encoder(x)
        
        layer_outputs = []
        
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
            layer_outputs.append(x)
        
        # Jump knowledge
        x = torch.cat(layer_outputs, dim=1)
        x = self.jump(x)
        
        # Pool
        x = global_mean_pool(x, batch)
        
        # Add global features
        if global_features is not None:
            global_encoded = self.global_processor(global_features)
            x = torch.cat([x, global_encoded], dim=1)
        
        # Decode
        out = self.decoder(x)
        
        # Enforce stochasticity
        if self.enforce_stochastic:
            out = F.softmax(out, dim=1)
        else:
            out = torch.sigmoid(out)
        
        return out
