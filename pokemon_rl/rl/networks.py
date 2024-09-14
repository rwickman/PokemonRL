import torch
import torch.nn as nn
import copy
import random
import numpy as np
from torch.optim import Adam
import torch.nn.functional as F
from torchvision.transforms import v2

print("np.random.randint(0, 5)", np.random.randint(0, 5))
print("random.random()", random.random())
print("torch.randn(5)", torch.randn(5))

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
        out_channels: list[int] = [32, 64, 64, 64],
        kernel_sizes=[7, 3, 3, 3],
        strides=[3, 2, 2, 2],
        linear_hidden_size: int = 32):

        super().__init__()
        feature_layers = []
        cur_in_channels = state_dim[0]

        assert len(out_channels) == len(kernel_sizes) == len(strides)

        for i in range(len(out_channels)):
            feature_layers.append(
                nn.Conv2d(cur_in_channels, out_channels[i], kernel_size=kernel_sizes[i], stride=strides[i], padding=1, bias=False))
            # feature_layers.append(nn.BatchNorm2d(out_channels[i]))
            feature_layers.append(nn.LeakyReLU())
            cur_in_channels = out_channels[i]

        # feature_layers.append(nn.Flatten())
        self.features = nn.Sequential(*feature_layers)

        # Get the output size of the CNN
        with torch.no_grad():
            cnn_out_dim = int(self.features(torch.randn(1, *state_dim)).view(1, -1).shape[-1])
            print("cnn_out_dim", cnn_out_dim)

        self.out = nn.Sequential(
            nn.Linear(cnn_out_dim, linear_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(linear_hidden_size, action_dim)
        )

        self.dynamics_model = TransitionModel(
            out_channels[-1], action_dim
        )

        self.online_proj = nn.Linear(cnn_out_dim, linear_hidden_size)
        self.spr_head = nn.Linear(linear_hidden_size, linear_hidden_size)

        feature_params = sum(p.numel() for p in self.features.parameters())
        out_params = sum(p.numel() for p in self.out.parameters())
        print(f"feature_params {feature_params} out_params {out_params} total_params {feature_params + out_params}")

    def state_pred(self, state, action=None, is_target=False):
        z = self.features(state)
        if is_target:
            return self.online_proj(z.view(state.shape[0], -1))
        else:
            z_t = self.dynamics_model(z, action).view(state.shape[0], -1)
            z_t = self.online_proj(z_t)
            return self.spr_head(z_t)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).view(x.shape[0], -1)
        x = self.out(x)
        return x


class TransitionModel(nn.Module):
    def __init__(self, channels: int, num_actions: int, num_layers: int = 2):
        super().__init__()
        self.num_actions = num_actions
         
        # layers = []
        # in_channels = channels + num_actions
        self.net = nn.Sequential(
            nn.Conv2d(channels + num_actions, channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(channels, channels),
            nn.LeakyReLU(),
    
            nn.Conv2d(channels, channels, 3, stride=1, padding=1)
        )
        # for i in range(layers):
        #     if i > 0:
        #         in_channels = channels
        #     layers.append(nn.Conv2d(in_channels, channels, 3, stride=1, padding=1))
        #     layers.append(nn.LeakyReLU())
            
        #     layers.append(nn.BatchNorm2d(channels, channels))
        # self.net = nn.Sequential(layers)

    def forward(self, x, action):
        batch_range = torch.arange(action.shape[0], device=action.device)
        action_onehot = torch.zeros(action.shape[0],
                                    self.num_actions,
                                    x.shape[-2],
                                    x.shape[-1],
                                    device=action.device)
        action_onehot[batch_range, action, :, :] = 1
        stacked_imgs = torch.cat([x, action_onehot], 1)
        next_state_pred = self.net(stacked_imgs)
        return next_state_pred

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
        self.state_dim = state_dim
        if len(state_dim) == 1:
            self.q_net = MPLValueNetwork(state_dim, action_dim)
        else:
            self.q_net = CNNValueNetwork(state_dim, action_dim)
            self.transforms = v2.Compose([
                v2.RandomResizedCrop(size=(state_dim[1], state_dim[2]), antialias=True),
                v2.ColorJitter(0.05, 0.05, 0.05, 0.05)
            ])
        self.q_net_tgt = copy.deepcopy(self.q_net)
        self.optimizer = Adam(self.q_net.parameters(), lr=self.args.lr)

    def compute_spr_loss(self, state_seq, actions_seq):
        spr_loss = 0
        for i in range(self.args.spr_k):
            next_state_emb_pred = self.q_net.state_pred(state_seq[i], actions_seq[i])
            next_state_emb = self.q_net_tgt.state_pred(
                self.transforms(state_seq[i+1].reshape(-1, 1, *self.state_dim[1:])).reshape(*state_seq[i+1].shape), is_target=True)

            next_state_emb_pred = F.normalize(next_state_emb_pred, p=2., dim=-1, eps=1e-3)
            next_state_emb = F.normalize(next_state_emb, p=2., dim=-1, eps=1e-3)
            spr_loss += F.mse_loss(next_state_emb_pred, next_state_emb, reduction="none").sum(-1).mean(0)
        return spr_loss
    
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

    def get_val_preds(self, states, actions, rewards, next_states, dones, n_steps):
        # Get the Q-values for the actions
        q_vals_matrix = self.q_net(states)
        q_vals = q_vals_matrix.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(dim=-1)
            
            # Get the next actions            
            q_next_target  = self.q_net_tgt(next_states)

            # Compute the td-targets using Double Q-Learning
            dqn_target = inverse_rescaling(q_next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1))
            
            if self.args.continuous:
                td_targets = rewards + self.args.gamma ** n_steps * dqn_target
            else:
                td_targets = rewards + self.args.gamma ** n_steps * dqn_target * (1.0 - dones)
    
        return q_vals, rescaling(td_targets)

    def update(self, states, action_seq, rewards, next_states, state_seq, dones, n_steps):
        q_vals, td_targets = self.get_val_preds(states, action_seq[0], rewards, next_states, dones, n_steps)
        if len(self.state_dim) > 1 and self.args.spr_lam > 0.0:
            spr_loss = self.compute_spr_loss(state_seq, action_seq)
        else:
            spr_loss = 0

        q_val_loss = F.mse_loss(q_vals, td_targets)
        loss = q_val_loss + self.args.spr_lam * spr_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        self._update_target()
        self._update_iter += 1
        if self._update_iter % 1024 == 0:
            print("loss", loss, "q_val_loss", q_val_loss, "spr_loss", spr_loss)
            print("n_steps", n_steps)
            print("rewards", rewards)
            print("dones", dones)
            print("q_vals", q_vals)
            print("td_targets", td_targets)
            print("actions ", action_seq[0])


        return loss

# m = CNNValueNetwork(1, (1, 72, 80), 10)
# x = torch.randn(5, 1, 72, 80)
# y = m(x)

# print(y.shape)