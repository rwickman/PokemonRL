import torch
import random


from pokemon_rl.rl.networks import DQN
from pokemon_rl.rl.replay_buffer import ReplayBuffer, MixedBuffer


class DQNAgent:
    def __init__(self, args, state_dim, action_dim):
        self.args = args
        self.train_dict = {
            "num_episodes": 0
        }
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.dqn = DQN(self.args, state_dim, action_dim).to(self.device)
        self.replay_buffer = MixedBuffer(state_dim, action_dim, self.args.replay_buffer_size, n_step=self.args.n_step, spr_k=self.args.spr_k, gamma=self.args.gamma, num_episodes=self.args.buffer_episodes)

    @property
    def epsilon(self):
        return max(self.args.eps * self.args.eps_decay ** self.train_dict["num_episodes"], self.args.eps_min) 

    def __call__(self, state, evaluate=False):
        with torch.no_grad():
            should_sample = not evaluate and random.random() <= self.epsilon 
            actions = self.dqn(torch.tensor(state).to(self.device).unsqueeze(0), should_sample)
            return actions[0]

    def add_trajectory(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def add_episode(self, ep_dict):
        self.replay_buffer.add_episode(ep_dict, sum(ep_dict["rewards"]))

    def update(self):
        state, action_seq, reward, next_state, state_seq, done, n_steps = self.replay_buffer.sample(self.args.batch_size)
        loss = self.dqn.update(state, action_seq, reward, next_state, state_seq, done, n_steps)
        return loss

    def mark_episode(self):
        self.train_dict["num_episodes"] += 1