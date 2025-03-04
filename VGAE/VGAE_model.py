from typing import Tuple

import torch
from torch import nn
from torch_geometric.data import Data


class Encoder(nn.Module):
    def __init__(self, hidden_model, mean_model, std_model, use_edge_attr: bool = False):
        super().__init__()

        self.hidden_model = hidden_model
        self.mean_model = mean_model
        self.std_model = std_model
        self.use_edge_attr = use_edge_attr

    def encode(self,
               x: torch.Tensor,
               edge_index: torch.LongTensor,
               edge_attr: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:

        hidden = self.hidden_model(x, edge_index, edge_attr)
        mean = self.mean_model(hidden, edge_index, edge_attr)
        std = self.std_model(hidden, edge_index, edge_attr)

        return mean, std

    def forward(self, X,adj,edge_attr):
        x, edge_index, edge_attr = X,adj,None
        mu, logvar = self.encode(x, edge_index, edge_attr)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, activation=torch.sigmoid, dropout: float = 0.1):
        super().__init__()

        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.dropout(z)
        adj_reconstructed = torch.matmul(z, z.T)

        if self.training:
            adj_reconstructed = self.activation(adj_reconstructed)

        return adj_reconstructed


class VGAE(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def reparametrize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std) + mu
        else:
            return mu

    def forward(self, X,adj):
        mu, logvar = self.encoder(X,adj,None)
        z = self.reparametrize(mu, logvar)
        adj = self.decoder(z)
        return adj, mu, logvar
