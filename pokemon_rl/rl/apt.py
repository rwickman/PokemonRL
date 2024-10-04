import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

class RMS:
    def __init__(self, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape).to(device)
        self.S = torch.ones(shape).to(device)
        self.n = epsilon

    def __call__(self, x):
        with torch.no_grad():
            bs = x.size(0)
            delta = torch.mean(x, dim=0) - self.M
            new_M = self.M + delta * bs / (self.n + bs)
            new_S = (self.S * self.n + torch.var(x, dim=0) * bs + (delta**2) * self.n * bs / (self.n + bs)) / (self.n + bs)

            self.M = new_M
            self.S = new_S
            self.n += bs

            return self.M, self.S

def compute_apt_reward(source, target, args, rms):
    with torch.no_grad():
        b1, b2 = source.size(0), target.size(0)
        # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
        sim_matrix = torch.norm(source[:, None, :].view(b1, 1, -1) - target[None, :, :].view(1, b2, -1), dim=-1, p=2)

        reward, _ = sim_matrix.topk(args.knn_k, dim=1, largest=False, sorted=True)  # (b1, k)

        reward = reward.reshape(-1, 1)  # (b1 * k, 1)

        moving_mean, moving_std = rms(reward)
        reward = reward / moving_std

        reward = torch.max(reward - args.knn_clip, torch.zeros_like(reward).to(device))
        reward = reward.reshape((b1, args.knn_k))  # (b1, k)
        reward = reward.mean(dim=1)  # (b1,)
        reward = torch.log(reward + 1.0)
        return reward.detach()