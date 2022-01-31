import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class BNN(nn.Module):
    """A Bayesian Neural Network with a CNN-type functional model.
    conv_layout: (in_channels, out_channels, kernel_size, stride, padding): tuple,
        (kernel_size, stride): tuple # for pooling layer
    linear_layout:  in_features: int
    For classical FCN, simply do not add conv layers to the layout.
    """
    def __init__(self, conv_layout: list, linear_layout: list, wpriors: list, bpriors: list):
        super().__init__()
        self.wpriors = wpriors
        self.bpriors = bpriors

        self.layers = [] # Combines both conv and linear layers
        all_layers = [] # Combines all layers (together with activation layers, ...)
        # Conv and pool layers (feature learning)
        for conv_layer, pool_layer in zip(conv_layout[::2], conv_layout[1::2]):
            conv = nn.Conv2d(*conv_layer)
            pool = nn.MaxPool2d(*pool_layer)
            self.layers.append(conv)
            all_layers.extend([conv, pool, nn.ReLU(inplace=True)])
        all_layers.append(nn.Flatten()) # Flatten input to fit into FCN
        # Linear layers (classification)
        for i, in_features in enumerate(linear_layout[:-1]):
            out_features = linear_layout[i+1]
            fc = nn.Linear(in_features, out_features)
            if i == len(linear_layout)-2:
                act = nn.LogSoftmax(dim=1) # Log likelihood
            else:
                act = nn.ReLU(inplace=True)
            self.layers.append(fc)
            all_layers.extend([fc, act])
        self.nn = nn.Sequential(*all_layers)
        self.__reset_weights_biases()

    def forward(self, x: torch.Tensor):
        return self.nn(x)

    def weights_biases(self):
        """Returns the weight and bias tensors.
        """
        weights, biases = [], []
        for layer in self.layers:
            weights.append(layer.weight)
            biases.append(layer.bias)
        return weights, biases

    def __reset_weights_biases(self):
        """Resets weights and biases according to the priors.
        """
        for i, layer in enumerate(self.layers):
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