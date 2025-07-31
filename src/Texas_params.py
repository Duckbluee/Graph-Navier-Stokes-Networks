import argparse
"""

"""
best_params_dict = {
'texas': { 'alpha_dim':'sc', 'beta_dim':'vc', 'beta_diag':True,
          'method':'rk4', 'time': 4.97935,'step_size': 1,
          'epoch':200, 'lr': 0.020534 ,'decay': 0.0058551,
          'block':'constant', 'hidden_dim': 128 , 'data_norm':'rw', 'self_loop_weight':1,
          'input_dropout': 0.35795, 'dropout': 0.64698, 'c': 0.1, 'count': 1, 'delta': 0.73885, 'hg': 0.11,
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
