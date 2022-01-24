from torch.nn.parameter import Parameter
import torch.nn as nn

class FCModel(nn.Module):
    def __init__(self, layout: list, wpriors: list, bpriors: list):
        super().__init__()
        self.wpriors = wpriors
        self.bpriors = bpriors

        self.linear_layers = []
        all_layers = []
        for i, in_features in enumerate(layout[:-1]):
            out_features = layout[i+1]
            fc = nn.Linear(in_features, out_features)
            if i == len(layout)-2:
                act = nn.LogSoftmax(dim=1) # produces log likelihoods
            else:
                act = nn.ReLU(inplace=True)
            self.linear_layers.append(fc)
            all_layers.extend([fc, act])
        self.nn = nn.Sequential(*all_layers)
        self.__reset_weights_biases()

    def forward(self, x):
        return self.nn(x)

    def weights_biases(self):
        weights, biases = [], []
        for layer in self.linear_layers:
            weights.append(layer.weight)
            biases.append(layer.bias)
        return weights, biases

    def __reset_weights_biases(self):
        for i, layer in enumerate(self.linear_layers):
            layer.weight = Parameter(self.wpriors[i].sample_like(layer.weight))
            layer.bias = Parameter(self.bpriors[i].sample_like(layer.bias))

    def logprior(self):
        """Calculate the log prior directly from this model's parameters.
        Since the priors are considered independent, we have to sum up the
        log priors.
        """
        loglike = 0
        weights, biases = self.weights_biases()
        for w, prior in zip(weights, self.wpriors):
            loglike += prior.loglike(w).sum()
        for b, prior in zip(biases, self.bpriors):
            loglike += prior.loglike(b).sum()
        return loglike

        


