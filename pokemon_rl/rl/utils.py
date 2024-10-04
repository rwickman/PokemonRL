import torch
import torch.nn as nn
import torch.nn.functional as F

def create_mlp(
        inp_size,
        out_size,
        n_layers,
        hidden_size,
        use_spectral_norm=False,
        spectral_ceof=1.0,
        hidden_w_init=nn.init.xavier_normal_,
        hidden_b_init=nn.init.zeros_,
        bias=True):

    layers = []
    
    # Add the hidden layers
    cur_inp_size = inp_size
    for i in range(n_layers):
        if use_spectral_norm:
            linear_layer = spectral_norm(
                    nn.Linear(cur_inp_size, hidden_size, bias=bias), spectral_coef=spectral_ceof) 
        else:
            linear_layer = nn.Linear(cur_inp_size, hidden_size, bias=bias)

        # Initialize the weights
        hidden_w_init(linear_layer.weight)
        if bias:
            hidden_b_init(linear_layer.bias)

        layers.append(linear_layer)
        layers.append(nn.ReLU(inplace=True))
        cur_inp_size = hidden_size

    # Add the output layer    
    if use_spectral_norm:
        out_layer = spectral_norm(
                nn.Linear(hidden_size, out_size), spectral_coef=spectral_ceof)
    else:
        out_layer = nn.Linear(hidden_size, out_size, bias=bias)

    # Initialize the weights
    hidden_w_init(out_layer.weight)
    if bias:
        hidden_b_init(out_layer.bias)

    layers.append(out_layer)

    return nn.Sequential(*layers)
