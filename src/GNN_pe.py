import torch
import torch.nn.functional as F
from base_classes import BaseGNN
from model_configurations import set_block, set_function
from torch import nn
from torch.nn import Parameter


# Define the GNN model.
class GNN(BaseGNN):
  def __init__(self, opt, dataset, device=torch.device('cpu')):
    super(GNN, self).__init__(opt, dataset, device)
    self.f = set_function(opt)
    block = set_block(opt)

    self.nodes_num = dataset.data.x.shape[0]

    mean = self.T
    node_t = torch.relu(0.5*torch.randn(self.nodes_num)+mean).to(device)

    time_tensor = torch.tensor([0, max(node_t+1)]).to(device)
    self.odeblock = block(self.f, self.regularization_fns, opt, dataset.data, device, node_t, t=time_tensor).to(device)
    self.odeblock.odefunc.GNN_postXN = self.GNN_postXN
    self.odeblock.odefunc.GNN_m2 = self.m2

    self.trusted_mask = None


  def encoder(self, x, pos_encoding=None):
    # Encode each node based on its feature.
    self.nodes_num = x.size(0)
    self.hyp_dim = x.size(1)
    c = self.opt['c']

    self.hyperbolic_embeddings = nn.Parameter(self.linear(x))
    ogx = self.linear1(x)
    norm = torch.norm(self.hyperbolic_embeddings, dim=1, keepdim=True)
    hyperbolic_embeddings = self.hyperbolic_embeddings / (norm * (1 + torch.sqrt(1 + c * norm ** 2)))
    x = torch.cat([x, hyperbolic_embeddings], dim=1)

    x = self.m1(x)

    if self.opt['use_mlp']==True:
      x = F.dropout(x, self.opt['dropout'], training=self.training)
      x = F.dropout(x + self.m11(F.relu(x)), self.opt['dropout'], training=self.training)
      x = F.dropout(x + self.m12(F.relu(x)), self.opt['dropout'], training=self.training)

    return x



  def forward_XN(self, x, pos_encoding=None):
    ###forward XN
    x = self.encoder(x, pos_encoding)
    self.odeblock.set_x0(x)
    if self.opt['function']=='gnsn':
      if self.opt['beta_diag'] == True:
        self.odeblock.odefunc.Beta = self.odeblock.odefunc.set_Beta()
    z = self.odeblock(x)
    return z

  def GNN_postXN(self, z):
    if self.opt['augment']==True:
      z = torch.split(z, z.shape[1] // 2, dim=1)[0]
    # Activation.
    if self.opt['XN_activation']==True:
      z = F.relu(z)
    # fc from bottleneck
    if self.opt['fc_out']==True:
      z = self.fc(z)
      z = F.relu(z)
    # Dropout.
    z = F.dropout(z, self.opt['dropout'], training=self.training)
    return z

  def forward(self, x, pos_encoding=None):
    z = self.forward_XN(x, pos_encoding)
    z = self.GNN_postXN(z)
    # Decode each node embedding to get node label.
    z = self.m2(z)
    return z