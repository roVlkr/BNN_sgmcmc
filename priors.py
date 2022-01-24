import torch

class Normal:
    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma
        self.sigma2 = sigma**2

    def loglike(self, theta: torch.Tensor):
        return -theta.view(-1).sub(self.mu).square_().div_(2*self.sigma2)

    def sample_like(self, theta: torch.Tensor):
        return self.sigma * torch.randn_like(theta).cuda() + self.mu

class Laplace:
    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma
        self.torch_dist = torch.distributions.Laplace(mu, sigma)

    def loglike(self, theta: torch.Tensor):
        return -theta.view(-1).sub(self.mu).abs_().div_(self.sigma)

    def sample_like(self, theta: torch.Tensor):
        return self.torch_dist.rsample(theta.shape)

class SoftUniform:
    def __init__(self, low: float, high: float, flatness: float=50):
        self.low = low
        self.high = high
        self.flatness = flatness

    def loglike(self, theta: torch.Tensor):
        """Multiply two sigmoid functions
        """
        return 1 / torch.exp(-self.flatness*(theta-self.low)).add(1)\
            / torch.exp(self.flatness*(theta-self.high)).add(1)

    def sample_like(self, theta: torch.Tensor):
        return torch.rand_like(theta) * (self.high - self.low) + self.low
