import torch

class Gamma:
    def __init__(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta

    @torch.no_grad()
    def dLogLike(self, theta: torch.Tensor):
        return (self.alpha - 1) / theta - self.beta

class Normal:
    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma
        self.sigma2 = sigma**2

    @torch.no_grad()
    def dLogLike_neg(self, theta: torch.Tensor): # negate for minimizing algorithm
        return (theta - self.mu) / self.sigma2

    def sample_like(self, theta: torch.Tensor):
        return self.sigma * torch.randn_like(theta).cuda() + self.mu

    def __add__(self, p):
        return Normal(self.mu + p.mu, self.sigma + p.sigma)

    def __mul__(self, alpha: float):
        return Normal(self.mu * alpha, self.sigma * alpha)
