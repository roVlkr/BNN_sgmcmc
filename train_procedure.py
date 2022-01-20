from multiprocessing.sharedctypes import Value
from urllib.parse import non_hierarchical
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from torch.utils.data import BatchSampler, RandomSampler
import math

import time

import optimizer
from model import Model

# This module makes the bayesian inference

class ParamRecorder:
    def __init__(self, max_records: int):
        self.records = []
        self.max_records = max_records
        self.current = None

    def save(self, h: float, params: dict):
        """We save (h_{k+1}, params_k) for Chen15 avg"""
        if self.current is not None:
            self.records.append((h, self.current))
            if len(self.records) > self.max_records:
                self.records.pop(0)
        self.current = params

class Training:
    BURN_IN = 50 # Number of iterations in the burn-in phase

    def __init__(self, train_set, test_set, model: Model, batch_size: int, opt_params: dict):
        self.train_set = train_set
        self.test_set = test_set
        self.N_train = len(train_set)
        self.N_test = len(test_set)
        self.batch_size = batch_size
        self.model = model
        self.optim = {
            'sgld': optimizer.SGLD(model.parameters(), model.priors, opt_params),
            'psgld': optimizer.pSGLD(model.parameters(), model.priors, opt_params),
            'msgld': optimizer.MSGLD(model.parameters(), model.priors, opt_params),
            'asgld': optimizer.ASGLD(model.parameters(), model.priors, opt_params)
        }[opt_params['name']]
        self.criterion = nn.CrossEntropyLoss(reduction='mean') # combines LogSoftmax layer and NLLLoss
        self.param_recorder = ParamRecorder(200)

    def start(self, thinning: int=20, epochs: int=5):
        losses, test_err = [], []
        k = 0
        agg_lr = 0

        for e in range(epochs):
            sampler = BatchSampler(RandomSampler(self.train_set, replacement=True), self.batch_size, True)
            num_batches = len(sampler)
            for b, batch in enumerate(sampler):
                x = torch.stack([self.train_set[i][0] for i in batch]).cuda().view(self.batch_size, -1)
                y = torch.tensor([self.train_set[i][1] for i in batch]).cuda()                
                loss = self.__optim(x, y) # Optimize params

                agg_lr += self.optim.lr() # 'time difference' between samples
                if k > Training.BURN_IN and (k+1) % thinning == 0:
                    self.param_recorder.save(agg_lr, self.model.state_dict())
                    agg_lr = 0
                self.__print_progressbar(e+1, b+1, num_batches)
                k += 1 # iteration
            losses.append(loss)
            # Get test accuracy
            err = self.__test_error()
            print(f'Iteration {k}: Test error {err:0.4f}')
            test_err.append(err)
        return self.param_recorder.records, losses, test_err

    def __optim(self, x, y):
        # Propagate forward
        y_pred = self.model(x)
        loss = self.N_train * self.criterion(y_pred, y)

        # Backpropagation
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.detach().item()

    @torch.no_grad()
    def __test_error(self):
        err = 0
        batch_size = 500
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, pin_memory=True)
        for x, y in test_loader:
            x, y = x.cuda(non_blocking=True).view(batch_size, -1), y.cuda(non_blocking=True)
            y_pred = torch.argmax(self.model(x), dim=1)
            err += torch.sum(y_pred != y) / self.N_test
        return err.item()

    def __print_progressbar(self, epoch, b, num_batches):
        progress = b / num_batches
        bar = math.ceil(progress*100/5)
        print(f'\rEpoch {epoch}: Batch {b}/{num_batches} [{("=" * int(bar)) + (" " * int(20-bar))}]', end='')
        if b == num_batches:
            print()

class Evaluation:
    def __init__(self, test_set, model: Model, param_records: list, batch_size: int):
        self.test_loader = DataLoader(test_set, batch_size=batch_size, pin_memory=True)
        self.N_test = len(test_set)
        self.param_records = param_records
        self.model = model
        self.batch_size = batch_size

    @torch.no_grad()
    def eval_current_state(self, x: torch.Tensor): # classical evaluation
        return torch.argmax(F.softmax(self.model(x), dim=1), dim=1)

    @torch.no_grad()
    def eval(self, x: torch.Tensor): # bayesian evaluation
        T = sum([h for (h, _) in self.param_records])
        avg = 0
        for h, param in self.param_records:
            self.model.load_state_dict(param)
            avg += h * F.softmax(self.model(x), dim=1) / T # Chen15 average
        arg_max = torch.argmax(avg, dim=1)
        max_prob = torch.max(avg, dim=1)
        return arg_max, max_prob

    def eval_all(self):
        acc = 0
        acc_just = 0
        for x, y in self.test_loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            x_flat = x.reshape(self.batch_size, -1)
            arg_max, max_prob = self.eval(x_flat)
            acc += torch.sum(y == arg_max) / self.N_test
        return acc.item(), acc_just#.item()