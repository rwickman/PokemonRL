
import numpy as np
import torch


class ReplayBuffer:
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0
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


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.tensor(self.state[ind]).to(self.device),
			torch.tensor(self.action[ind]).to(self.device),
			torch.tensor(self.reward[ind]).to(self.device),
			torch.tensor(self.next_state[ind]).to(self.device),
			torch.tensor(self.done[ind]).to(self.device)
		)

	def sample_states(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)
		return torch.tensor(self.state[ind]).to(self.device)