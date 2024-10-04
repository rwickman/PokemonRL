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
from pokemon_rl.rl.apt import RMS, compute_apt_reward

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

class CNNTrunk(nn.Module):
    def __init__(
        self,
        state_dim,
        out_channels: list = [32, 64, 64, 64],
        kernel_sizes=[8, 3, 3, 3],
        strides=[2, 2, 2, 2],
        padding=[0, 0, 1, 1]):
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

        self.trunk = nn.Sequential(*feature_layers)

        with torch.no_grad():
            self.cnn_out_dim = self.trunk(torch.randn(1, *state_dim))
            print("self.cnn_out_dim none flat shape", self.cnn_out_dim.shape)
            self.cnn_out_dim  = int(self.cnn_out_dim.view(1, -1).shape[-1])
            print("self.cnn_out_dim", self.cnn_out_dim)

    def forward(self, state):
        return self.trunk(state).view(state.shape[0], -1)
        

class CNNValueNetwork(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim: int = 128):
        super().__init__()
        self.trunk = CNNTrunk(state_dim)

        self.head = MLPValueNetwork((self.trunk.cnn_out_dim, ), action_dim, hidden_dim, 1)

    def get_state_latents(self, state):
        state_features = self.trunk(state)
        state_latent = self.head.trunk(state_features)
        state_adv_latent = self.head.advantage_trunk(state_latent)
        return state_latent, state_adv_latent

    def forward_with_inter(self, state):
        return self.head.forward_with_inter(self.trunk(state))

    def advantage(self, state):
        return self.head.advantage(self.trunk(state))

    def forward(self, state):
        return self.head(self.trunk(state))


class CNNActor(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim: int = 128,
        n_layers: int = 2):
        super().__init__()
        self.trunk = CNNTrunk(state_dim)
        self.head = MLPNetwork((self.trunk.cnn_out_dim, ), action_dim, hidden_dim, n_layers)

    def forward_with_inter(self, state):
        return self.head.forward_with_inter(self.trunk(state))

    def forward(self, state):
        return self.head(self.trunk(state))


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
    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dim: int = 128,
                 n_layers: int = 2):
        super().__init__()
        hidden_layers = [
            nn.Linear(*state_dim, hidden_dim),
            nn.ReLU()
        ] 
        for i in range(n_layers - 1):
            hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            if i + 1 < n_layers - 1:
                hidden_layers.append(nn.ReLU())

        self.trunk = nn.Sequential(*hidden_layers)

        self.advantage_trunk = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.advantage_head = nn.Sequential(
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
        state_adv_latent = self.advantage_trunk(state_latent)
        advantage = self.advantage_head(state_adv_latent)
        q_val = value + advantage - advantage.mean(dim=-1, keepdim=True)

        return q_val, state_latent, state_adv_latent

    def advantage(self, state):
        return self.advantage_head(self.advantage_trunk(self.trunk(state)))

    def forward(self, state):
        state_latent = self.trunk(state)
        value = self.value_head(state_latent)
        advantage = self.advantage_head(self.advantage_trunk(state_latent))
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
            if i + 1 < n_layers - 1:
                hidden_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*hidden_layers)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward_with_inter(self, state):
        state_latent = self.trunk(state)
        return self.head(state_latent), state_latent

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
        self.rms = RMS()

        print("self.temp_dec_per_step: ", self.temp_dec_per_step)
        print("state_dim", state_dim)
        print("action_dim", action_dim)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._init_networks()
        self.gamma_sums = torch.zeros(self.args.n_step, device=self.device) 
        self.gamma_sums[0] = 1
        for i in range(1, self.args.n_step):
            self.gamma_sums[i] = self.args.gamma ** i + self.gamma_sums[i-1]
        print(self.gamma_sums)

    def _init_networks(self):
        if len(self.state_dim) == 1:
            self.q_net = MLPValueNetwork(self.state_dim, self.action_dim, self.args.hidden_dim, self.args.n_layers).to(self.device)
            self.actor = MLPNetwork(self.state_dim, self.action_dim, self.args.hidden_dim, self.args.n_layers).to(self.device)

        else:
            self.actor = CNNActor(self.state_dim, self.action_dim, self.args.hidden_dim, self.args.n_layers).to(self.device)
            self.q_net = CNNValueNetwork(self.state_dim, self.action_dim, hidden_dim=self.args.hidden_dim).to(self.device)
            # self.transforms = v2.Compose([
            #     v2.RandomResizedCrop(size=(self.state_dim[1], self.state_dim[2]), antialias=True),
            #     v2.ColorJitter(0.05, 0.05, 0.05, 0.05)
            # ])
       

        self._state_disc = StateDiscriminator(self.args.hidden_dim, self.args.hidden_dim).to(self.device)
        self._state_adv_disc= StateDiscriminator(self.args.hidden_dim, self.args.hidden_dim).to(self.device)
        self._state_actor_disc = StateDiscriminator(self.args.hidden_dim, self.args.hidden_dim).to(self.device)

        self.q_net_tgt = copy.deepcopy(self.q_net)
        # TODO: FORGOT TO ADD self._state_adv_disc
        self.optimizer = Adam(
            list(self.q_net.parameters()) + list(self._state_disc.parameters()) + list(self._state_adv_disc.parameters()),
            lr=self.args.lr
        )
        self.actor_optimizer = Adam(
            list(self.actor.parameters()) + list(self._state_actor_disc.parameters()),
            lr=self.args.actor_lr
        )

    def reset_networks(self):
        self.actor_temp = self.actor_temp + self.args.reset_temp_offset + self.temp_dec_per_step * self.args.ramp_up_updates
        old_nets = [self.q_net, self.q_net_tgt, self._state_disc, self._state_adv_disc, self._state_actor_disc, self.actor]
        self._init_networks()
        new_nets = [self.q_net, self.q_net_tgt, self._state_disc, self._state_adv_disc, self._state_actor_disc, self.actor]

        # Restore some of the target network parameters  
        for old_net, new_net in zip(old_nets, new_nets):
            for tgt_param, param in zip(new_net.parameters(), old_net.parameters()):
                tgt_param.data.copy_(
                    self.args.reset_tgt_tau * param.data + (1.0 - self.args.reset_tgt_tau) * tgt_param.data)


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
        q_vals_matrix, state_latent, state_adv_latent = self.q_net.forward_with_inter(states)
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
    
        return q_vals, rescaling(td_targets), state_latent, state_adv_latent

    def update(self, states, actions, rewards, next_states, dones, n_steps):
        did_reset = False

        if self._update_iter > 0 and self._update_iter % self.args.network_reset_step == 0:
            print("\nRESETTING\n")
            self.reset_networks()
            did_reset = True

        if len(self.state_dim) == 1:
            next_states_latent = self.q_net.trunk(next_states)
        else:
            next_states_latent, next_states_adv_latent = self.q_net.get_state_latents(next_states)
            # print("next_states_latent", next_states_latent[:3, :5])
            # print("next_states_adv_latent", next_states_adv_latent[:3, :5])

        apt_reward = compute_apt_reward(next_states_latent, next_states_latent, self.args, self.rms)
        # Scale to account for n-step
        apt_reward = apt_reward * self.gamma_sums[n_steps-1]

        q_vals, td_targets, state_latent, state_adv_latent = self.get_val_preds(states, actions, rewards + self.args.apt_reward_lam * apt_reward, next_states, dones, n_steps)

        if len(self.state_dim) == 1:
            next_states_adv_latent = self.q_net.advantage_trunk(self.q_net.trunk(next_states))


        # if len(self.state_dim) > 1 and self.args.spr_lam > 0.0:
        #     spr_loss = self.compute_spr_loss(state_seq, action_seq)
        # else:
        #     spr_loss = 0

        q_val_loss = F.mse_loss(q_vals, td_targets)
        contrastive_loss = self._state_disc.contrastive_loss(state_latent, next_states_latent)
        contrastive_loss_adv = self._state_adv_disc.contrastive_loss(state_adv_latent, next_states_adv_latent)

        loss = q_val_loss + self.args.contrastive_lam * contrastive_loss + self.args.contrastive_lam * contrastive_loss_adv
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        self._update_target()
        self._update_iter += 1

        if self._update_iter % 4096 == 0:
            print("\nloss", loss, "q_val_loss", q_val_loss, "contrastive_loss", contrastive_loss)
            print("n_steps", n_steps)
            print("rewards", rewards)
            print("apt_reward", apt_reward)
            print("dones", dones)
            print("q_vals", q_vals)
            print("td_targets", td_targets)
            print("actions ", actions)
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
        return loss, actor_loss, contrastive_loss, did_reset

    def update_actor(self, state, next_states):
        with torch.no_grad():
            q_vals = self.q_net.advantage(state)
            # max_q_val, min_q_val = q_vals.max(dim=-1), q_vals.min(dim=-1)

            # q_vals = (q_vals - min_q_val) / (max_q_val -  min_q_val)
            # q_vals_norm = q_vals / q_vals.mean(dim=0)
            action_distr = (q_vals /self.args.q_val_dist_temp).softmax(dim=-1)

        actor_preds, state_latent = self.actor.forward_with_inter(state)
        if len(self.state_dim) == 1:
            next_states_latent = self.actor.trunk(next_states)
        else:
            next_states_latent = self.actor.head.trunk(self.actor.trunk(next_states))

        if self._update_iter % 4096 == 0:
            print("q_vals", q_vals[:5])
            print("QVAL action_distr:", action_distr[:5], action_distr.shape)
            print("actor_preds DISTR:", actor_preds.softmax(dim=-1)[:5])
            print("actor_preds WITH TEMP DISTR:", (actor_preds/self.actor_temp).softmax(dim=-1)[:5])
            print("self.actor_temp", self.actor_temp, "\n")

        contrastive_loss = self._state_actor_disc.contrastive_loss(state_latent, next_states_latent)

        actor_loss = F.cross_entropy(actor_preds, action_distr) 
        total_loss =  actor_loss + self.args.contrastive_lam * contrastive_loss

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