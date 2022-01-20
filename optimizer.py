import math
import torch
from torch.optim.optimizer import Optimizer
from typing import Iterable

from abc import ABC, abstractmethod

class SGMCMC(Optimizer, ABC):
    """opt_params = dict(name, lr0, min_lr, gamma, ...)
    """
    def __init__(self, params: Iterable, priors: Iterable, opt_params: dict):
        defaults = dict(priors=priors, **opt_params)
        super(SGMCMC, self).__init__(params, defaults)

    @abstractmethod
    def reset_variables(self, param, state):
        pass
    
    @abstractmethod
    def apply_alg(self, param, grad, prior, group, state):
        pass

    def lr(self):
        return next(iter(self.state.values()))['lr']

    @torch.no_grad()
    def step(self):
        params_with_grad, priors, grads = [], [], []
        
        for group in self.param_groups:
            lr0, gamma, min_lr = group['lr0'], group['gamma'], group['min_lr']
            
            for param, prior in zip(group['params'], group['priors']):
                if param.grad is not None:
                    params_with_grad.append(param)
                    grads.append(param.grad)
                    priors.append(prior)

                    # Dynamically adjust learning rate
                    state = self.state[param]
                    if len(state) == 0:
                        self.reset_variables(param, state)
                        state['lr'], state['step'] = lr0, 1
                    if state['lr'] > min_lr:
                        # polynomial decay
                        state['lr'] = max(lr0 * state['step']**(-gamma), min_lr)
                    state['step'] += 1

            for param, grad, prior in zip(params_with_grad, grads, priors):
                self.apply_alg(param, grad, prior, group, self.state[param])


class SGLD(SGMCMC):
    """Implements Stochastic Gradient Langevin Dynamics. This Algorithm is based on the implementation of the
    SGD-Algorithm from the PyTorch library (https://pytorch.org/docs/stable/_modules/torch/optim/adam.html).
    Standard value for gamma=1/3 (see Chen15 MSE bound)

    lr = lr0 * k^(-gamma)
    opt_params: lr0, gamma, min_lr, tau
    """

    def __init__(self, params: Iterable, priors: Iterable, opt_params: dict):
        super(SGLD, self).__init__(params, priors, opt_params)

    def reset_variables(self, param, state):
        pass

    def apply_alg(self, param, grad, prior, group, state):
        # Get variables
        tau = group['tau']
        lr = state['lr']

        # SGLD-Algorithm
        Z = torch.randn_like(grad).cuda()
        grad.add_(prior.dLogLike_neg(param)) # grad log likelihood + grad log prior
        param.add_(grad, alpha=-lr/2)\
            .add_(Z, alpha=math.sqrt(lr * tau)) # noise


class pSGLD(SGMCMC):
    """See above
    Default parameters see Lietal16 and Chen15

    opt_params: lr0, min_lr, gamma, alpha, eps, tau
    """

    def __init__(self, params: Iterable, priors: Iterable, opt_params: dict):
        super(pSGLD, self).__init__(params, priors, opt_params)

    def reset_variables(self, param, state):
        state['V'] = torch.zeros_like(param).cuda()

    def apply_alg(self, param, grad, prior, group, state):
        # Get variables
        alpha, eps, tau = group['alpha'], group['eps'], group['tau']
        lr, V = state['lr'], state['V']

        # pSGLD-Algorithm
        Z = torch.randn_like(grad).cuda()
        V.mul_(alpha).addcmul_(grad, grad, value=1-alpha) # vector format, no diagonal matrix
        D = V.sqrt().add_(eps)

        # Due to diagonal form (originally) -> elementwise products
        grad.add_(prior.dLogLike_neg(param))
        param.addcdiv_(grad, D, value=-lr/2)\
            .addcmul_(D.sqrt(), Z, value=math.sqrt(lr * tau))


class MSGLD(SGMCMC):
    """    opt_params: lr0, min_lr, gamma, beta1, a, tau
    """

    def __init__(self, params: Iterable, priors: Iterable, opt_params: dict):
        super(MSGLD, self).__init__(params, priors, opt_params)

    def reset_variables(self, param, state):
        state['m'] = torch.zeros_like(param).cuda()
        state['gradU'] = torch.zeros_like(param).cuda()

    def apply_alg(self, param, grad, prior, group, state):
        # Get variables
        beta1, a, tau = group['beta1'], group['a'], group['tau']
        lr, m = state['lr'], state['m']
        gradU = state['gradU']

        # MSGLD-Algorithm
        Z = torch.randn_like(grad).cuda()
        m.mul_(beta1).add_(gradU, alpha=1-beta1)
        gradU = grad.add_(prior.dLogLike_neg(param))
        param.add_(gradU + a*m, alpha=-lr/2)\
            .add_(Z, alpha=math.sqrt(lr * tau))


class ASGLD(SGMCMC):
    """    opt_params: lr0, min_lr, gamma, beta1, beta2, a, eps, tau
    """

    def __init__(self, params: Iterable, priors: Iterable, opt_params: dict):
        super(ASGLD, self).__init__(params, priors, opt_params)

    def reset_variables(self, param, state):
        state['m'] = torch.zeros_like(param).cuda()
        state['V'] = torch.zeros_like(param).cuda()
        state['gradU'] = torch.zeros_like(param).cuda()

    def apply_alg(self, param, grad, prior, group, state):
        # Get variables
        beta1, beta2 = group['beta1'], group['beta2']
        eps, a, tau = group['eps'], group['a'], group['tau']
        lr, m = state['lr'], state['m']
        gradU, V = state['gradU'], state['V']

        # ASGLD-Algorithm
        Z = torch.randn_like(grad).cuda()
        m.mul_(beta1).add_(gradU, alpha=1-beta1)
        V.mul_(beta2).addcmul_(gradU, gradU, value=1-beta2)
        D = V.sqrt().add_(eps)
        gradU = grad.add_(prior.dLogLike_neg(param))
        param.add_(gradU + a*m.div(D), alpha=-lr/2)\
            .add_(Z, alpha=math.sqrt(lr * tau))