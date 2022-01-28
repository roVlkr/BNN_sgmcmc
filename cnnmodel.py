from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch

class CNNModel(nn.Module):
    def __init__(self, layout: list, wpriors: list, bpriors: list):
        super().__init__()
        self.wpriors = wpriors
        self.bpriors = bpriors

        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,10)


    def forward(self, X: torch.Tensor):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 5*5*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)

    def weights_biases(self):
        pass

    def __reset_weights_biases(self):
        pass

    def logprior(self):
        pass

if __name__ == '__main__':
    model = CNNModel(None, None, None)
    print(model.conv1.weight.shape)