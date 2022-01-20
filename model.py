import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.parallel import DataParallel

class Model(nn.Module):
    def __init__(self, num_features: int, num_labels: int, num_hidden: list, priors: object):
        super().__init__()
        # Initialize layers and priors for their weights and biases
        layer_szs = [num_features] + num_hidden + [num_labels]
        if type(priors) is not list:
            priors_ = []
            for _ in layer_szs: # appends priors in the order of the parameters (first weight, then bias)
                priors_.extend([priors, priors]) # weight_prior, bias_prior 
            priors = priors_ # has now twice as much priors as there are layers

        num_layers = len(layer_szs)-1
        layers = []
        for i in range(num_layers):
            layer = nn.Linear(in_features=layer_szs[i], out_features=layer_szs[i+1])

            # Adjust weights and biases according to prior
            weight_prior, bias_prior = priors[2*i], priors[2*i+1]
            layer.weight = Parameter(weight_prior.sample_like(layer.weight))
            layer.bias = Parameter(bias_prior.sample_like(layer.bias))

            layers.append(DataParallel(layer))
            if i < num_layers-1:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.BatchNorm1d(layer_szs[i+1]))
        self.nn = nn.Sequential(*layers)
        self.priors = priors
    
    def forward(self, x):
        return self.nn(x)

    def param_distribution(self, num_buckets=20):
        # Gather information about the parameters
        param_list = []
        min_value, max_value = float('inf'), float('-inf')
        for layer in self.nn:
            try: # "Normal" layer
                w, b = layer.module.weight, layer.module.bias
                param_list.extend([w, b])
                min_value = min([w.min().item(), b.min().item(), min_value])
                max_value = max([w.max().item(), b.max().item(), max_value])
            except:
                pass
        # Calculate distribution
        space = torch.linspace(start=min_value, end=max_value, steps=num_buckets+1)
        buckets = torch.zeros(size=(num_buckets,))
        for param in param_list:
            for bucket, (interval_min, interval_max) in enumerate(zip(space[:-1], space[1:])):
                test = torch.logical_and(param >= interval_min, param < interval_max)
                buckets[bucket] += param[test].numel()
        buckets = buckets / buckets.sum() # norm to make sum = 1
        space = 0.5 * (space[1:] + space[:-1]) # take intermediate values
        return space, buckets