from cmath import sqrt
import math
import torch
from torch.optim.optimizer import Optimizer
from typing import Iterable

from abc import ABC, abstractmethod

class SGMCMC(Optimizer, ABC):
    """opt_params = dict(name, lr0, min_lr, gamma, ...)
    lr = max(lr0 * k^(-gamma), min_lr)

    Implementations are based on Adam-Algorithm from the PyTorch library
    (https://pytorch.org/docs/stable/_modules/torch/optim/adam.html)
    """
    def __init__(self, params: Iterable, opt_params: dict):
        self.opt_params = opt_params
        super(SGMCMC, self).__init__(params, opt_params)

    @torch.no_grad()
    def save_gradlikelihood(self):
        """Save only the likelihood part of the gradient but also some
        of the required information for the step to complete.
        """
        # Gather all required information for one step
        self.current_step = {
            'params': [], # params with grad
            'gradlikelihoods': [], # gradient without prior
            'gradUs': [] } # gradient with prior
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    grad = param.grad.detach().clone() # make deep copy
                    x = grad.norm()
                    self.current_step['params'].append(param)
                    self.current_step['gradlikelihoods'].append(grad)

    @torch.no_grad()
    def step(self):
        lr0 = self.opt_params['lr0']
        min_lr = self.opt_params['min_lr']
        gamma = self.opt_params['gamma']

        # Now save the whole gradient (gradU, after save_gradlikelihood)
        for param in self.current_step['params']:
            self.current_step['gradUs'].append(param.grad)

            # Dynamically adjust learning rate
            state = self.state[param]
            if len(state) == 0:
                self.reset_variables(param, state)
                state['lr'], state['step'] = lr0, 1
            if state['lr'] > min_lr:
                # polynomial decay
                state['lr'] = max(lr0 * state['step']**(-gamma), min_lr)

        # Apply SGMCMC algorithm
        for i, param in enumerate(self.current_step['params']):
            grad = self.current_step['gradlikelihoods'][i]
            gradU = self.current_step['gradUs'][i]
            state = self.state[param]
            self.apply_alg(param, grad, gradU, state)
            state['step'] += 1

    @abstractmethod
    def reset_variables(self, param, state):
        pass
    
    @abstractmethod
    def apply_alg(self, param, grad, gradU, state):
        pass

    def lr(self):
        return next(iter(self.state.values()))['lr']


class SGLD(SGMCMC):
    """opt_params: lr0, gamma, min_lr, tau
    """

    def __init__(self, params: Iterable, opt_params: dict):
        super(SGLD, self).__init__(params, opt_params)

    def reset_variables(self, param, state):
        pass

    def apply_alg(self, param, grad, gradU, state):
        # Get variables
        tau = self.opt_params['tau']
        lr = state['lr']

        # SGLD-Algorithm
        Z = torch.randn_like(grad).cuda()
        param.add_(gradU, alpha=-lr/2)\
            .add_(Z, alpha=math.sqrt(lr * tau)) # noise


class pSGLD(SGMCMC):
    """opt_params: lr0, min_lr, gamma, alpha, eps, tau, N
    """

    def __init__(self, params: Iterable, opt_params: dict):
        super(pSGLD, self).__init__(params, opt_params)

    def reset_variables(self, param, state):
        state['V'] = torch.zeros_like(param).cuda()

    def apply_alg(self, param, grad, gradU, state):
        # Get variables
        alpha = self.opt_params['alpha']
        eps = self.opt_params['eps']
        tau = self.opt_params['tau']
        N = self.opt_params['N']
        lr, V = state['lr'], state['V']

        # pSGLD-Algorithm
        Z = torch.randn_like(grad).cuda()
        mean_grad = grad/N
        V.mul_(alpha).addcmul_(mean_grad, mean_grad, value=1-alpha) # vector format, no diagonal matrix
        G = 1/V.sqrt().add_(eps)

        # Due to diagonal form (originally) -> elementwise products
        param.addcmul_(gradU, G, value=-lr/2)\
            .addcmul_(Z, G.sqrt(), value=math.sqrt(lr * tau))


class MSGLD(SGMCMC):
    """opt_params: lr0, min_lr, gamma, beta1, a, tau
    """

    def __init__(self, params: Iterable, opt_params: dict):
        super(MSGLD, self).__init__(params, opt_params)

    def reset_variables(self, param, state):
        state['m'] = torch.zeros_like(param).cuda()

    def apply_alg(self, param, grad, gradU, state):
        # Get variables
        beta1 = self.opt_params['beta1']
        a = self.opt_params['a']
        tau = self.opt_params['tau']
        lr, m = state['lr'], state['m']

        # MSGLD-Algorithm
        Z = torch.randn_like(grad).cuda()
        param.add_(gradU.add(m, alpha=a), alpha=-lr/2)\
            .add_(Z, alpha=math.sqrt(lr * tau))
        m.mul_(beta1).add_(gradU, alpha=1-beta1)


class ASGLD(SGMCMC):
    """opt_params: lr0, min_lr, gamma, beta1, beta2, a, eps, tau
    """

    def __init__(self, params: Iterable, opt_params: dict):
        super(ASGLD, self).__init__(params, opt_params)

    def reset_variables(self, param, state):
        state['m'] = torch.zeros_like(param).cuda()
        state['V'] = torch.zeros_like(param).cuda()

    def apply_alg(self, param, grad, gradU, state):
        # Get variables
        beta1, beta2 = self.opt_params['beta1'], self.opt_params['beta2']
        eps, a = self.opt_params['eps'], self.opt_params['a']
        tau = self.opt_params['tau']
        lr, m, V = state['lr'], state['m'], state['V']

        # ASGLD-Algorithm
        Z = torch.randn_like(grad).cuda()
        D_inv = 1/V.add(eps).sqrt_()
        param.add_(D_inv.mul_(m).mul_(a).add_(gradU), alpha=-lr/2)\
            .add_(Z, alpha=math.sqrt(lr * tau))
        m.mul_(beta1).add_(gradU, alpha=1-beta1)
        V.mul_(beta2).addcmul_(gradU, gradU, value=1-beta2)


class SGHMC(SGMCMC):
    """opt_params: lr0, min_lr, gamma, M, C
    """
    def __init__(self, params: Iterable, opt_params: dict):
        super(SGHMC, self).__init__(params, opt_params)

    def reset_variables(self, param, state):
        state['xi'] = torch.zeros_like(param).cuda() # momentum variable

    def apply_alg(self, param, grad, gradU, state):
        # Get variables
        M, C = self.opt_params['M'], self.opt_params['C']
        lr, xi = state['lr'], state['xi']

        Z = torch.randn_like(grad).cuda()
        # SGHMC algorithm (symmetric splitting integrator)
        xi.add_(gradU, alpha=-lr).add_(Z, alpha=math.sqrt(2*C*lr))
        param.add_(xi, alpha=lr/(2*M))
        xi.mul_(math.exp(-C*lr/M)) # two steps together
        param.add_(xi, alpha=lr/(2*M))


class SGRMC(SGMCMC):
    """opt_params: lr0, min_lr, gamma, m, c, C
    """
    def __init__(self, params: Iterable, opt_params: dict):
        super(SGRMC, self).__init__(params, opt_params)

    def reset_variables(self, param, state):
        state['xi'] = torch.zeros_like(param)

    def apply_alg(self, param, grad, gradU, state):
        m, c = self.opt_params['m'], self.opt_params['c']
        C = self.opt_params['C']
        lr, xi = state['lr'], state['xi']

        # Relativistic mass
        M = xi.pow(2).sum().div_(c**2).add_(m**2).sqrt_().item()

        Z = torch.randn_like(grad).cuda()
        # Euler integrator
        xi.add_(gradU, alpha=-lr)\
            .add_(xi, alpha=-C*lr/M)\
            .add_(Z, alpha=math.sqrt(2*C*lr))
        param.add_(xi, alpha=lr/M)
