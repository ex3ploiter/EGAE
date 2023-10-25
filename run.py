from data_loader import *
from model import EGAE, GAE
import torch
import warnings
from torch_geometric.data import Data
import argparse


warnings.filterwarnings('ignore')


if __name__ == '__main__':
    
    name = 'cora'
    features, adjacency, labels = load_data(name)
    layers = [256, 128]
    acts = [torch.nn.functional.relu] * len(layers)
    # acts = [None, torch.nn.functional.relu]
    learning_rate = 10**-4*4
    pretrain_learning_rate = 0.001
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--Likelihood", type=str)
    
    
    
    
    args = parser.parse_args()
    
    
    
    
    
    Dataset_pyG = Data(x=torch.tensor(features).cuda(), edge_index=torch.tensor(adjacency).nonzero().t().contiguous().cuda(),y=torch.tensor(labels).cuda()).cuda()
    
    for coeff_reg in [0.001]:
        for alpha in [0.01, 0.1, 1, 10, 100]:
            print('========== alpha={}, reg={} =========='.format(alpha, coeff_reg))
            gae = EGAE(features, adjacency, labels, alpha, layers=layers, acts=acts,
                       max_epoch=50, max_iter=4, coeff_reg=coeff_reg, learning_rate=learning_rate,dataset=Dataset_pyG,LikelihoodFlag=bool(args.Likelihood))
            gae.pretrain(10, learning_rate=pretrain_learning_rate)
            losses = gae.run()
            scio.savemat('losses_{}.mat'.format(name), {'losses': np.array(losses)})
