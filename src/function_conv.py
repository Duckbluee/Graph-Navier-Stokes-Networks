import torch
from torch import nn
import torch_sparse
from torch.nn.init import uniform, xavier_uniform_
from base_classes import ODEFunc
from utils import MaxNFEException


class ODEFuncGNSN(ODEFunc):

    def __init__(self, in_features, out_features, opt, data, device):
        super(ODEFuncGNSN, self).__init__(opt, data, device)

        self.in_features = in_features
        self.out_features = out_features

        self.reaction_tanh = False
        if opt['beta_diag'] == True:
            self.b_W = nn.Parameter(torch.Tensor(in_features))
            self.reset_parameters()
        self.epoch = 0

        self.alpha_sc = nn.Parameter(0.0 * torch.ones(1, opt['hidden_dim']))
        self.beta_sc = nn.Parameter(-10.0 * torch.ones(1, opt['hidden_dim']))
        self.delta = opt['delta']

    def reset_parameters(self):
        if self.opt['beta_diag'] == True:
            uniform(self.b_W, a=-1, b=1)

    def set_Beta(self, T=None):
        Beta = torch.diag(self.b_W)
        return Beta

    def sparse_multiply(self, x):

        if self.opt['block'] in ['attention']:
            mean_attention = self.attention_weights.mean(dim=1)
            ax = torch_sparse.spmm(self.edge_index, mean_attention-self.delta, x.shape[0], x.shape[0], x)
            if self.opt['count']==1:
                ax1 = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[0], x.shape[0], x)
            elif self.opt['count']==2:
                ax1 = torch_sparse.spmm(self.edge_index, mean_attention * mean_attention, x.shape[0], x.shape[0], x)
            else:
                ax1 = torch_sparse.spmm(self.edge_index, mean_attention * mean_attention * mean_attention, x.shape[0], x.shape[0], x)
        else:
            edge_index = self.edge_index.to(torch.int64)
            ax = torch_sparse.spmm(edge_index, self.edge_weight-self.delta, x.shape[0], x.shape[0], x)
            if self.opt['count']==1:
                ax1 = torch_sparse.spmm(self.edge_index, self.edge_weight, x.shape[0], x.shape[0], x)
            elif self.opt['count']==2:
                ax1 = torch_sparse.spmm(self.edge_index, self.edge_weight * self.edge_weight, x.shape[0], x.shape[0], x)
            else:
                ax1 = torch_sparse.spmm(self.edge_index, self.edge_weight * self.edge_weight * self.edge_weight, x.shape[0], x.shape[0], x)
        return ax, ax1

    def forward(self, t, state):

        x, t_node = state

        mask = (t <= t_node).float().unsqueeze(-1)

        if self.nfe > self.opt["max_nfe"]:
            raise MaxNFEException

        self.nfe += 1
        if not self.opt['no_alpha_sigmoid']:
            alpha = torch.sigmoid(self.alpha_train)
            beta = torch.sigmoid(self.beta_train)
        else:
            alpha = self.alpha_train
            beta = self.beta_train
        ax, ax1 = self.sparse_multiply(x)
        diffusion = (ax - x)*self.opt['hg']

        conv = ax1 * mask * (1-self.opt['hg'])

        if self.opt['beta_diag'] == False:
            if self.opt['reaction_term'] == 'fb':
                f = alpha * diffusion + beta * conv
            elif self.opt['reaction_term'] == 'fb3':
                f = alpha * diffusion + beta * (conv + x)
            else:
                f = alpha * diffusion + beta * conv
        elif self.opt['beta_diag'] == True:
            f = alpha * diffusion + conv @ self.Beta

        if self.opt['add_source']:
            f = f + self.source_train * self.x0


        return f, torch.zeros_like(t_node)
