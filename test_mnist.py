import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib.pyplot as plt

from train_procedure import Training, Evaluation
from model import Model
from priors import Normal
from optimizer import *
from torchvision import datasets, transforms

def test_sgmcmc(train_set, test_set, opt_params, epochs, batch_size, thinning):
    print(f'Optimizer: {opt_params}')
    model = Model(num_features=784, num_labels=10, num_hidden=[400, 400], priors=Normal(0, 1000)).cuda()
    training = Training(train_set, test_set, model, batch_size, opt_params)
    param_records, losses, test_err = training.start(thinning, epochs)
    evaluation = Evaluation(test_set, model, param_records, batch_size)
    acc, acc_just = evaluation.eval_all()
    print(f'Bayes acc: {acc}, Bayes justified acc: {acc_just}')
    return losses, test_err

if __name__ == '__main__':
    # Define Data
    train_set = datasets.MNIST('./Data/MNIST', train=True, download=True, transform=transforms.ToTensor())
    test_set = datasets.MNIST('./Data/MNIST', train=False, download=True, transform=transforms.ToTensor())

    # Define optimizers
    opt_params = [
        #dict(name='sgld', lr0=1e-4, min_lr=1e-5, gamma=0.333, tau=1e-5),
        #dict(name='psgld', lr0=1e-4, min_lr=1e-5, gamma=0.333, alpha=0.99, eps=1e-8, tau=1e-5),
        dict(name='msgld', lr0=1e-4, min_lr=1e-5, gamma=0.333, beta1=0.99, a=0.3, tau=1e-5),
        #dict(name='asgld', lr0=1e-4, min_lr=1e-5, gamma=0.333, beta1=0.99, beta2=0.999, eps=1e-8, a=1, tau=1e-5)
    ]

    # Gather accuracy results of each optimizer
    test_errs = []
    train_losses = []
    thinning = 50
    for op in opt_params:
        losses, test_err = test_sgmcmc(train_set, test_set, opt_params=op, batch_size=256, epochs=10, thinning=thinning)
        test_errs.append(test_err)
        train_losses.append(train_losses)
    
    for err in test_errs:
        plt.plot([i*thinning for i in range(len(err))], err)
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.legend([op['name'] for op in opt_params])
    plt.show()

    
    
    

    