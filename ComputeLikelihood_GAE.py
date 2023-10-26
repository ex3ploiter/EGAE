from VGAE.VGAE_model import VGAE, Encoder, Decoder
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import Subset

from tqdm.notebook import tqdm

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import Sequential, GCNConv

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from VGAE.VGAE_utils import adj_matrix_from_edge_index
from main_utils import get_VGAE_hidden_models
from VGAE.VGAE_loss import VGAELoss


def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD

class LikelihoodComputer(nn.Module):
    def __init__(self, X,adj):
        super(LikelihoodComputer, self).__init__()
        self.config = config = {
            "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
            "LR": 0.001,
            "EPOCHS": 10 , # Adjust the number of epochs as needed
            "hidden_dim" : 64
        }
        self.device = self.config["DEVICE"]
        self.X = X
        self.adj=adj


        # Define loss function and optimizer
        self.loss_function = VGAELoss(norm=2)
        self.optimizer = Adam(params=self.model.parameters(), lr=config["LR"])

        self.train()

    def train_epoch(self):
        
        self.model.train()
        self.optimizer.zero_grad()
        recovered, mu, logvar = self.model(self.X, self.adj)
        loss = loss_function(preds=recovered, labels=adj_label,
                             mu=mu, logvar=logvar, n_nodes=n_nodes,
                             norm=norm, pos_weight=pos_weight)
        loss.backward()
        cur_loss = loss.item()
        self.optimizer.step()

        hidden_emb = mu.data.numpy()
        




    def train(self):
        for epoch in range(self.config["EPOCHS"]):
            self.train_epoch()

    def ComputeLikelihood(self):
        adj_output, _, _ = self.model(self.X,self.adj)
        adj_output = nn.Sigmoid()(adj_output)
        return adj_output.mean()

    def forward(self):
        return self.ComputeLikelihood()