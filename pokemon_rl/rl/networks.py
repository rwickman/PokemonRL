import torch
import torch.nn as nn
from torch.distributions import Categorical
import copy
import random
import numpy as np
from torch.optim import Adam
import torch.nn.functional as F
from torchvision.transforms import v2

from pokemon_rl.rl.state_discriminator import StateDiscriminator 

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
        out_channels: list = [32, 64, 64, 64, 64],
        kernel_sizes=[8, 4, 3, 3, 3],
        strides=[4, 2, 2, 2, 2],
        padding=[0, 1, 1, 1, 1],
        linear_hidden_size: int = 64):

        super().__init__()
        feature_layers = []
        cur_in_channels = state_dim[0]

        assert len(out_channels) == len(kernel_sizes) == len(strides)

        for i in range(len(out_channels)):
            feature_layers.append(
                nn.Conv2d(cur_in_channels, out_channels[i], kernel_size=kernel_sizes[i], stride=strides[i], padding=padding[i], bias=False))
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
        # print("x.shape", x.shape)
        # for layer in self.features:
        #     x = layer(x)
        #     print("x.shape", x.shape)
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

class MLPValueNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim: int = 128, n_layers: int = 2):
        super().__init__()
        hidden_layers = [
            nn.Linear(*state_dim, hidden_dim),
            nn.ReLU()
        ] 
        for i in range(n_layers - 1):
            hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            if i + 1 < n_layers:
                hidden_layers.append(nn.ReLU())

        self.trunk = nn.Sequential(*hidden_layers)

        self.advantage_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.value_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )


    def forward_with_inter(self, state):
        state_latent = self.trunk(state)
        value = self.value_head(state_latent)
        advantage = self.advantage_head(state_latent)
        q_val = value + advantage - advantage.mean(dim=-1, keepdim=True)

        return q_val, state_latent

    def forward(self, state):
        state_latent = self.trunk(state)
        value = self.value_head(state_latent)
        advantage = self.advantage_head(state_latent)
        q_val = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_val



class MLPNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim: int = 128, n_layers: int = 2):
        super().__init__()
        hidden_layers = [
            nn.Linear(*state_dim, hidden_dim),
            nn.ReLU()
        ] 
        for i in range(n_layers - 1):
            hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            if i + 1 < n_layers:
                hidden_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*hidden_layers)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward_with_inter(self, state):
        state = state_latent = self.trunk(state)
        return self.head(state), state_latent

    def forward(self, state):
        return self.head(self.trunk(state))

class DQN(nn.Module):
    def __init__(self, args, state_dim, action_dim):
        super().__init__()
        self.args = args
        self._update_iter = 0
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.actor_temp = self.args.actor_temp_init
        self.temp_dec_per_step = (self.args.actor_temp_init - self.args.actor_temp_final) / self.args.actor_temp_decay_steps
        print("self.temp_dec_per_step: ", self.temp_dec_per_step)
        print("state_dim", state_dim)
        print("action_dim", action_dim)
        if len(state_dim) == 1:
            self.q_net = MLPValueNetwork(state_dim, action_dim, self.args.hidden_dim, self.args.n_layers)
            self.actor = MLPNetwork(state_dim, action_dim, self.args.hidden_dim, self.args.n_layers)
            self._state_disc = StateDiscriminator(self.args.hidden_dim, self.args.hidden_dim)
            self._state_actor_disc = StateDiscriminator(self.args.hidden_dim, self.args.hidden_dim)

        else:
            self.q_net = CNNValueNetwork(state_dim, action_dim)
            self.transforms = v2.Compose([
                v2.RandomResizedCrop(size=(state_dim[1], state_dim[2]), antialias=True),
                v2.ColorJitter(0.05, 0.05, 0.05, 0.05)
            ])
        self.q_net_tgt = copy.deepcopy(self.q_net)
        self.optimizer = Adam(
            list(self.q_net.parameters()) + list(self._state_disc.parameters()),
            lr=self.args.lr
        )
        self.actor_optimizer = Adam(
            list(self.actor.parameters()) + list(self._state_actor_disc.parameters()),
            lr=self.args.actor_lr
        )

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
        with torch.no_grad():
            action_prob = (self.actor(state)/self.actor_temp).softmax(dim=-1)
            if self.args.render_mode != None:
                print(action_prob)

            # print(self.q_net(state))

            distr = Categorical(action_prob)
            action = distr.sample()

            return action.cpu().numpy()
        # q_vals = self.q_net(state)
        # if should_sample:
        #     return np.random.randint(0, self.action_dim, size=state.shape[0])
        # else:
        #     with torch.no_grad():
        #         return q_vals.argmax(dim=-1).cpu().numpy()

    def get_val_preds(self, states, actions, rewards, next_states, dones, n_steps):
        # Get the Q-values for the actions
        q_vals_matrix, state_latent = self.q_net.forward_with_inter(states)
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
    
        return q_vals, rescaling(td_targets), state_latent

    def update(self, states, action_seq, rewards, next_states, state_seq, dones, n_steps):
        q_vals, td_targets, state_latent = self.get_val_preds(states, action_seq[0], rewards, next_states, dones, n_steps)
        next_states_latent = self.q_net.trunk(next_states)

        # if len(self.state_dim) > 1 and self.args.spr_lam > 0.0:
        #     spr_loss = self.compute_spr_loss(state_seq, action_seq)
        # else:
        #     spr_loss = 0

        q_val_loss = F.mse_loss(q_vals, td_targets)
        contrastive_loss = self._state_disc.contrastive_loss(state_latent, next_states_latent)        

        loss = q_val_loss + self.args.contrastive_lam * contrastive_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        self._update_target()
        self._update_iter += 1

        if self._update_iter % 1024 == 0:
            print("loss", loss, "q_val_loss", q_val_loss, "contrastive_loss", contrastive_loss)
            print("n_steps", n_steps)
            print("rewards", rewards)
            print("dones", dones)
            print("q_vals", q_vals)
            print("td_targets", td_targets)
            print("actions ", action_seq[0])
            # action_prob = (self.actor(states)/self.actor_temp).softmax(dim=-1)
            # # print(states[:2])
            # print("action_prob", action_prob[:10])


        # if self._update_iter % 8 == 0:
        #     # with torch.no_grad():
        #     #     for param in self.q_net_tgt.parameters():
        #     #         param.add_(torch.randn(param.size(), device=states.device) * 0.0001)

        #     with torch.no_grad():
        #         for param in self.actor.parameters():
        #             param.add_(torch.randn(param.size(), device=states.device) * 0.0001)

        actor_loss = self.update_actor(states, next_states)
        return loss, actor_loss, contrastive_loss

    def update_actor(self, state, next_states):
        with torch.no_grad():
            q_vals = self.q_net(state)
            max_q_val, min_q_val = q_vals.max(), q_vals.min()

            q_vals = (q_vals - min_q_val) / (max_q_val -  min_q_val)

            action_distr = (q_vals/0.2).softmax(dim=-1)

        actor_preds, state_latent = self.actor.forward_with_inter(state)
        next_states_latent = self.actor.trunk(next_states)
        if self._update_iter % 1024 == 0:
            print("QVAL action_distr:", action_distr[:6], action_distr.shape)
            print("actor_preds", actor_preds[:6], actor_preds.shape)
            print("actor_preds DISTR:", actor_preds.softmax(dim=-1)[:6])
            print("actor_preds WITH TEMP DISTR:", (actor_preds/self.actor_temp).softmax(dim=-1)[:6])

            print("self.actor_temp", self.actor_temp)

        contrastive_loss = self._state_actor_disc.contrastive_loss(state_latent, next_states_latent)

        actor_loss = F.cross_entropy(actor_preds, action_distr) 
        total_loss =  actor_loss + 0.1 * self.args.contrastive_lam * contrastive_loss

        self.actor_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()
        self.actor_temp = max(self.actor_temp - self.temp_dec_per_step, self.args.actor_temp_final)
        return actor_loss

# m = CNNValueNetwork(1, (1, 72, 80), 10)
# x = torch.randn(5, 1, 72, 80)
# y = m(x)

# print(y.shape)