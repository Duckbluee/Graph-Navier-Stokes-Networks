import argparse
"""

"""
best_params_dict = {
'Cora': { 'alpha_dim':'vc', 'beta_dim':'vc', 'beta_diag':True,
          'method':'rk4', 'time': 1.36018,'step_size':0.2,
          'epoch':300, 'lr': 0.0045227 ,'decay': 0.0077897,
          'block':'attention', 'hidden_dim': 64, 'data_norm':'gcn', 'self_loop_weight':0.5,
          'input_dropout': 0.32073, 'dropout': 0.5103,
          'use_mlp': False, 'm2_mlp': True, 'XN_activation': True, 'c': 0.2, 'count': 3, 'delta': 0.0012983, 'hg': 0.81,
           },
}

def shared_grand_params(opt):
    opt['block'] = 'constant'
    opt['function'] = 'laplacian'
    opt['optimizer'] = 'adam'
    opt['epoch'] = 200
    opt['lr'] = 0.001
    opt['method'] = 'euler'
    opt['geom_gcn_splits'] = True
    return opt

def shared_gnsn_params(opt):
    opt['function'] = 'gnsn'
    opt['optimizer'] = 'adam'
    opt['geom_gcn_splits'] = True
    return opt

def hetero_params(opt):
    #added self loops and make undirected for chameleon & squirrel
    if opt['dataset'] in ['chameleon', 'squirrel']:
        opt['hetero_SL'] = True
        opt['hetero_undir'] = True
    return opt
