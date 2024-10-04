
import numpy as np
import torch
import math
import random
from scipy.stats import betabinom

#77ff6f

def join_samples(sample_1, sample_2):
	for i in range(len(sample_1)):
	

		sample_1[i] = torch.cat((sample_1[i], sample_2[i]), dim=0)
	return sample_1

class MixedBuffer:
	def __init__(self, state_dim: tuple, action_dim: int, max_size=int(1e4), n_step: int = 1, spr_k: int = 5, gamma: float = 0.99, num_episodes: int = 20):
		self.replay_buffer = ReplayBuffer(state_dim, action_dim, max_size, n_step, spr_k, gamma)
		self.ep_buffer = EpisodeBuffer(state_dim, action_dim, max_size, n_step, spr_k, gamma, num_episodes)
		self._num_times_sampled = 0

	def add(self, state, action, reward, next_state, done):
		self.replay_buffer.add(state, action, reward, next_state, done)
	
	def add_episode(self, ep_dict: dict, ep_return: float):
		self.ep_buffer.add_episode(ep_dict, ep_return)
	
	def sample(self, batch_size):
		self._num_times_sampled += 1
		if self.ep_buffer.is_ready:
			replay_exps = self.replay_buffer.sample(batch_size//2, self._num_times_sampled)
			ep_exps = self.ep_buffer.sample(batch_size//2, self._num_times_sampled)
			return join_samples(replay_exps, ep_exps)
		else:
			return self.replay_buffer.sample(batch_size)
	
class EpisodeBuffer:
	def __init__(self, state_dim: tuple, action_dim: int, max_size=int(1e4), n_step: int = 1, spr_k: int = 5, gamma: float = 0.99, num_episodes: int = 20):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.num_episodes = num_episodes
		self.ptr = 0
		self.size = 0
		self.num_adds = 0
		self.ep_returns = [(i, -float(math.inf)) for i in range(self.num_episodes)]
		self.replay_buffers = [
			ReplayBuffer(self.state_dim, action_dim, 1, n_step, spr_k, gamma) for _ in range(num_episodes)]
	
	@property
	def is_ready(self):
		return self.size == self.num_episodes

	def add_episode(self, ep_dict: dict, ep_return: float):
		random.shuffle(self.ep_returns)
		self.ep_returns = sorted(self.ep_returns, key=lambda x: x[1])
		if self.ep_returns[0][1] <= ep_return:
			idx = self.ep_returns[0][0]
			
			# Replace episode with better experiences
			self.ep_returns[0] = (idx, ep_return)
			self.replay_buffers[idx].action = np.array(ep_dict["actions"])
			self.replay_buffers[idx].state = np.stack(ep_dict["states"])
			self.replay_buffers[idx].reward = np.array(ep_dict["rewards"], dtype=np.float32)
			self.replay_buffers[idx].next_state = np.stack(ep_dict["next_states"])
			self.replay_buffers[idx].done = np.array(ep_dict["dones"], dtype=np.float32)
			self.replay_buffers[idx].size = len(ep_dict["actions"])
			self.size = min(self.size + 1, self.num_episodes)

		if self.num_adds % 50 == 0:
			print(self.ep_returns)
		self.num_adds += 1

	def sample(self, batch_size: int, num_times_sampled: int = 0):
		exps = None
		small_batch_size = 8
		for i in range(batch_size//small_batch_size):
			ep_idx = random.randint(0, self.size-1)
			cur_exps = self.replay_buffers[ep_idx].sample(small_batch_size, num_times_sampled)
			if exps is None:
				exps = cur_exps
			else:
				exps = join_samples(cur_exps, exps) 
		return exps 

class ReplayBuffer:
	def __init__(self, state_dim: tuple, action_dim: int, max_size=int(1e5), n_step: int = 1, spr_k: int = 5, gamma: float = 0.99):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0
		self.n_step = n_step
		self.spr_k = spr_k
		self.gamma = gamma
		self.action_dim = action_dim
		self.state = np.zeros((max_size, *state_dim), dtype=np.float32)
		self.action = np.zeros(max_size, dtype=np.int64)
		self.reward = np.zeros(max_size, dtype=np.float32)
		self.next_state = np.zeros((max_size, *state_dim), dtype=np.float32)
		self.done = np.zeros(max_size, dtype=np.uint8)
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, reward, next_state, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.reward[self.ptr] = reward
		self.next_state[self.ptr] = next_state
		self.done[self.ptr] = done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size, num_times_sampled: int = 0):
		ind = np.random.randint(0, self.size, size=batch_size)
		n_steps = np.random.randint(1, self.n_step + 1, size=batch_size)
		n_steps = betabinom.rvs(self.n_step - 1, 1.0, min(1.0 + num_times_sampled * 0.00001, 10), size=batch_size) + 1

		n_step_ind = np.clip(ind + n_steps - 1, 0, self.size-1)
		# Recompute n-steps so that it accounts for the clipping
		n_steps = n_step_ind - ind + 1

		# state_seq = np.zeros((self.spr_k + 1, batch_size, *self.state[0].shape), dtype=self.state.dtype)
		# action_seq = np.zeros((self.spr_k + 1, batch_size), dtype=self.action.dtype)
		# for i in range(self.spr_k + 1):
		# 	cur_ind = np.clip(ind + i, 0, self.size-1)
		# 	state_seq[i] = self.state[cur_ind]
		# 	action_seq[i] = self.action[cur_ind]

		# Compute the n-step trajectories
		# rewards = np.zeros(batch_size, dtype=np.float32)
		# dones = np.zeros(batch_size, dtype=np.float32)
		# for i in range(batch_size):
		# 	for j in reversed(range(n_steps[i])):
		# 		cur_ind  = np.clip(ind + j, 0, self.size-1)
		# 		rewards[i] = self.reward[cur_ind][i] + self.gamma * rewards[i] * (1-self.done[cur_ind][i])
		# 		dones[i] = np.clip(self.done[cur_ind][i] + dones[i], 0, 1)
		
		rewards = self.reward[np.clip(ind + self.n_step - 1, 0, self.size-1)]
		dones = self.done[np.clip(ind + self.n_step - 1, 0, self.size-1)] * (n_steps >= self.n_step)


		for i in reversed(range(self.n_step - 1)):
			# (n_steps - 2 >= i) is 1 if the n-steps has been reached and 0 otherwise
			cur_ind  = np.clip(ind + i, 0, self.size-1)

			rewards = self.reward[cur_ind] + self.gamma * rewards * (1 - self.done[cur_ind]) * (n_steps - 2 >= i) 
			dones = np.clip(self.done[cur_ind] + dones, 0, 1) * (n_steps - 1 >= i)

		# if (np.round(rewards_1, 3) == np.round(rewards, 3)).sum() != batch_size:
		# 	print("rewards_1", rewards_1)
		# 	print("rewards", rewards)
		# 	print(rewards == rewards_1)
		# 	raise Exception("INVALID REWARDS")

		# if (dones_1 == dones).sum() != batch_size:
		# 	print("dones_1", dones_1)
		# 	print("dones", dones)
		# 	print("self.dones[ind]", self.done[ind])
		# 	raise Exception("INVALID DONES")

			# torch.tensor(self.next_state[ind + n_steps - 1]).to(self.device),
		return [
			torch.tensor(self.state[ind]).to(self.device),
			torch.tensor(self.action[ind]).to(self.device),
			torch.tensor(rewards).to(self.device),
			torch.tensor(self.next_state[n_step_ind]).to(self.device),
			torch.tensor(dones).to(self.device),
			torch.tensor(n_steps).to(self.device)
		]

	def sample_states(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)
		return torch.tensor(self.state[ind]).to(self.device)