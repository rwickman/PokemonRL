import torch
import torch.nn as nn
import copy
import random
import numpy as np
from torch.optim import Adam

def rescaling(x):
    """
    rescaling function
    https://github.com/google-research/seed_rl/blob/f53c5be4ea083783fb10bdf26f11c3a80974fa03/agents/r2d2/learner.py#L180
    Args:
      x (torch.tensor): input
    Returns:
      rescaled value
    """
    
    eps = 0.001
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.) - 1.) + eps * x
    # return x

def inverse_rescaling(x):
    """
    inverse rescaling function
    https://github.com/google-research/seed_rl/blob/f53c5be4ea083783fb10bdf26f11c3a80974fa03/agents/r2d2/learner.py#L186
    Args:
      x (torch.tensor): input
    Returns:
      inverse rescaled value
    """
    
    eps = 0.001
    return torch.sign(x) * (torch.square(((torch.sqrt(1. + 4. * eps * (torch.abs(x) + 1. + eps))) - 1.) / (2. * eps)) - 1.)

class CNNValueNetwork(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim: int,
        out_channels: list[int] = [32, 64, 64],
        kernel_sizes=[4, 3, 4],
        strides=[4, 2, 2],
        linear_hidden_size: int = 64):

        super().__init__()
        feature_layers = []
        cur_in_channels = state_dim[0]

        assert len(out_channels) == len(kernel_sizes) == len(strides)

        for i in range(len(out_channels)):
            feature_layers.append(
                nn.Conv2d(cur_in_channels, out_channels[i], kernel_size=kernel_sizes[i], stride=strides[i]))
            # feature_layers.append(nn.BatchNorm2d(out_channels[i]))
            feature_layers.append(nn.LeakyReLU(inplace=True))
            

            cur_in_channels = out_channels[i]
        feature_layers.append(nn.Flatten())
        self.features = nn.Sequential(*feature_layers)

        # Get the output size of the CNN
        with torch.no_grad():
            cnn_out_dim = int(self.features(torch.randn(1, *state_dim)).shape[-1])
            print("cnn_out_dim", cnn_out_dim)
        self.out = nn.Sequential(
            nn.Linear(cnn_out_dim, linear_hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(linear_hidden_size, action_dim)
        )

        feature_params = sum(p.numel() for p in self.features.parameters())
        out_params = sum(p.numel() for p in self.out.parameters())
        print(f"feature_params {feature_params} out_params {out_params} total_params {feature_params + out_params}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.out(x)
        return x

class MPLValueNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(*state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, state):
        return self.net(state)

class DQN(nn.Module):
    def __init__(self, args, state_dim, action_dim):
        super().__init__()
        self.args = args
        self._update_iter = 0
        self.action_dim = action_dim
        # self.q_net = MPLValueNetwork(state_dim, action_dim)
        self.q_net = CNNValueNetwork(state_dim, action_dim)
        self.q_net_tgt = copy.deepcopy(self.q_net)
        self.optimizer = Adam(self.q_net.parameters(), lr=self.args.lr)

    def _update_target(self):
        """Perform soft update of the target policy."""
        for tgt_param, param in zip(self.q_net_tgt.parameters(), self.q_net.parameters()):
            tgt_param.data.copy_(
                self.args.tgt_tau * param.data + (1.0 - self.args.tgt_tau) * tgt_param.data)

    def forward(self, state, should_sample=False):
        q_vals = self.q_net(state)
        if should_sample:
            return np.random.randint(0, self.action_dim, size=state.shape[0])
        else:
            with torch.no_grad():
                return q_vals.argmax(dim=-1).cpu().numpy()

    def get_val_preds(self, states, actions, rewards, next_states, dones):
        # Get the Q-values for the actions
        q_vals_matrix = self.q_net(states)
        q_vals = q_vals_matrix.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(dim=-1)

            # Get the next actions            
            q_next_target  = self.q_net_tgt(next_states)

            # Compute the td-targets using Double Q-Learning
            dqn_target = inverse_rescaling(q_next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1).detach())
            if self.args.continuous:
                td_targets = rewards + self.args.gamma * dqn_target
            else:
                td_targets = rewards + self.args.gamma * dqn_target * (1.0 - dones)

        return q_vals, rescaling(td_targets)

    def update(self, states, actions, rewards, next_states, dones):
        q_vals, td_targets = self.get_val_preds(states, actions, rewards, next_states, dones)
        td_errors = q_vals - td_targets
        loss = td_errors.pow(2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._update_target()
        self._update_iter += 1
        if self._update_iter % 1024 == 0:
            print("rewards", rewards)
            print("loss", loss)
            print("q_vals", q_vals)
            print("td_targets", td_targets)
            print("actions ", actions)

        return loss

# m = CNNValueNetwork(1, (1, 72, 80), 10)
# x = torch.randn(5, 1, 72, 80)
# y = m(x)

# print(y.shape)