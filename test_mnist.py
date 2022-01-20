import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib.pyplot as plt

from train_procedure import Training, Evaluation
from model import Model
from priors import Normal
from optimizer import *
from torchvision import datasets, transforms

def plot_distributions(prior, posterior):
    plt.fill_between(prior[0], prior[1], alpha=0.5, edgecolor='b')
    plt.fill_between(posterior[0], posterior[1], alpha=0.5, edgecolor='r')
    plt.xlabel('Parameter distributions')
    plt.legend(['Prior', 'Posterior'])
    plt.show()

def test_sgmcmc(train_set, test_set, opt_params, epochs, batch_size, thinning):
    print(f'Training configuration: Epochs {epochs}, Batch size {batch_size}, Thinning {thinning}')
    print(f'Optimizer: {opt_params}')
    
    param_dists = []    
    model = Model(num_features=784, num_labels=10, num_hidden=[400, 400], priors=Normal(0, 0.1)).cuda()
    prior = model.param_distribution() # record initial distribution
    
    training = Training(train_set, test_set, model, batch_size, opt_params)
    param_records, losses, test_err = training.start(epochs, thinning)
    posterior = model.param_distribution()

    evaluation = Evaluation(test_set, model, param_records, batch_size)
    #acc, acc_just = evaluation.eval_all()

    #print(f'Bayes acc: {acc}, Bayes justified acc: {acc_just}')
    plot_distributions(prior, posterior)
    return losses, test_err

if __name__ == '__main__':
    # Define Data
    train_set = datasets.MNIST('./Data/MNIST', train=True, download=True, transform=transforms.ToTensor())
    test_set = datasets.MNIST('./Data/MNIST', train=False, download=True, transform=transforms.ToTensor())

    # Define optimizers
    opt_params = [
        dict(name='sgld', lr0=1e-3, min_lr=1e-5, gamma=0.333, tau=1),
        dict(name='psgld', lr0=1e-3, min_lr=1e-5, gamma=0.333, alpha=0.99, eps=5e-2, tau=1),
        #dict(name='msgld', lr0=1e-4, min_lr=1e-5, gamma=0.333, beta1=0.99, a=1, tau=1),
        #dict(name='asgld', lr0=1e-4, min_lr=1e-5, gamma=0.333, beta1=0.99, beta2=0.999, eps=1e-8, a=1, tau=1)
    ]

    # Gather accuracy results of each optimizer
    test_errs = []
    train_losses = []
    epochs = 10
    batch_size = 128
    thinning = 50
    for op in opt_params:
        losses, test_err = test_sgmcmc(train_set, test_set, op, epochs, batch_size, thinning)
        test_errs.append(test_err)
        train_losses.append(train_losses)
    
    for err in test_errs:
        plt.plot([i*thinning for i in range(len(err))], err)
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.legend([op['name'] for op in opt_params])
    plt.show()

    
    
    

    