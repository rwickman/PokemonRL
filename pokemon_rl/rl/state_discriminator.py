import torch
import torch.nn as nn
import torch.nn.functional as F

from pokemon_rl.rl.utils import create_mlp


class StateDiscriminator(nn.Module):
    def __init__(self,
                 state_dim: int,
                 hidden_size: int,
                 temperature: float = 0.5):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.temperature = temperature
        
        self.num_iter = 0

        self._state_encoder = create_mlp(state_dim, state_dim, 1, hidden_size)
        
        self.loss_fn = nn.CrossEntropyLoss()

    def contrastive_loss(self, obs, obs_next):
        obs_latent = self._state_encoder(obs)
        obs_next_latent = self._state_encoder(obs_next)
        
        batch_size = obs_latent.shape[0]
        obs_latent = F.normalize(obs_latent, dim=1)
        obs_next_latent = F.normalize(obs_next_latent, dim=1)
        sim_matrix = torch.matmul(obs_latent, obs_next_latent.T)

        # Extract the positive and negative pairs
        mask = torch.eye(batch_size, dtype=torch.bool, device=self.device)
        positives = sim_matrix[mask.bool()].view(batch_size, -1)

        negatives = sim_matrix[~mask.bool()].view(batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1) / self.temperature
        labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        loss = self.loss_fn(logits, labels)

        return loss
