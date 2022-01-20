import torch.nn as nn

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
            # weight_prior, bias_prior = priors[2*i], priors[2*i+1]
            # layer.weight = Parameter(weight_prior.sample_like(layer.weight))
            # layer.bias = Parameter(bias_prior.sample_like(layer.bias))

            layers.append(nn.DataParallel(layer))
            if i < num_layers-1:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.BatchNorm1d(layer_szs[i+1]))
        self.nn = nn.Sequential(*layers)
        self.priors = priors
    
    def forward(self, x):
        return self.nn(x)

    def get_weight_distribution(self):
        for layer in self.layers:

