import os
from xml.etree.ElementInclude import DEFAULT_MAX_INCLUSION_DEPTH
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import matplotlib.pyplot as plt

from train_procedure import Training, Evaluation
from fcmodel import FCModel
from priors import *
from optimizer import *
import utils
from torchvision import datasets, transforms

# Must be in the same order as the trainings
OPTIMIZER_NAMES = ['SGLD', 'pSGLD', 'MSGLD', 'ASGLD', 'SGHMC', 'SGRMC']

# One Training of a SGMCMC optimizer
def test_sgmcmc(train_set, test_set, opt_params, epochs, batch_size, burn_in, thinning):
    print(f'Training configuration: Epochs {epochs}, Batch size {batch_size}, Thinning {thinning}')
    print(f'Optimizer: {opt_params}')
    
    network_layout = [784, 400, 100, 10]
    # ~ Xavier init
    wpriors=[Laplace(0, math.pow(n, -0.5)) for n in network_layout[:-1]]
    bpriors=[Laplace(0, math.pow(n, -1)) for n in network_layout[:-1]]
    model = FCModel(network_layout, wpriors, bpriors).cuda()
    weights, biases = model.weights_biases() # record prior params
    wpriors = [w.detach().cpu().view(-1) for w in weights]
    bpriors = [b.detach().cpu().view(-1) for b in biases]

    training = Training(train_set, test_set, model, batch_size, burn_in, opt_params)
    param_records, test_err = training.start(epochs, thinning)
    weights, biases = model.weights_biases() # record posterior params
    wposts = [w.detach().cpu().view(-1) for w in weights]
    bposts = [b.detach().cpu().view(-1) for b in biases]

    evaluation = Evaluation(test_set, model, param_records, batch_size=1000)
    acc = evaluation.eval_all()
    print(f'Accuracy of {opt_params["name"]}: {acc}')

    #utils.plot_distributions(wpriors, bpriors, wposts, bposts)
    return test_err, acc


####################################################################
# Fully connected Network test (without normalization of data)
def fc1_test(epochs, batch_size, burn_in, thinning):
    transform = transforms.ToTensor()
    train_set = datasets.MNIST('./Data/MNIST', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./Data/MNIST', train=False, download=True, transform=transform)

    # Define optimizers
    opt_params = [
        # Langevin dynamics
        dict(name='sgld', lr0=5e-5, min_lr=1e-6, gamma=0.333, tau=1),
        dict(name='psgld', lr0=1e-5, min_lr=1e-7, gamma=0.333, alpha=0.99, eps=5e-2, tau=1, N=len(train_set)),
        dict(name='msgld', lr0=5e-5, min_lr=1e-6, gamma=0.333, beta1=0.99, a=1, tau=1),
        dict(name='asgld', lr0=5e-5, min_lr=1e-6, gamma=0.333, beta1=0.99, beta2=0.9999, eps=1e-1, a=1, tau=1),

        # Hamiltonian dynamics
        dict(name='sghmc', lr0=1e-3, min_lr=1e-5, gamma=0.2, M=0.1, C=50),
        dict(name='sgrmc', lr0=5e-3, min_lr=1e-6, gamma=0.333, m=1, c=100, C=100)  
    ]

    # Gather accuracy results of each optimizer
    test_errs = []
    test_accs = []
    for op in opt_params:
        err, acc = test_sgmcmc(train_set, test_set, op, epochs, batch_size, burn_in, thinning)
        test_errs.append(err)
        test_accs.append(acc)
    return np.array(test_errs), np.array(test_accs)


####################################################################
# Fully connected network test (with normalization of data)
def fc2_test(epochs, batch_size, burn_in, thinning):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.1307, std=0.3081)
    ])
    train_set = datasets.MNIST('./Data/MNIST', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./Data/MNIST', train=False, download=True, transform=transform)

        # Define optimizers
    opt_params = [
        # Langevin dynamics
        #dict(name='sgld', lr0=2e-5, min_lr=5e-7, gamma=0.333, tau=1),
        dict(name='psgld', lr0=1e-6, min_lr=5e-8, gamma=0.333, alpha=0.999, eps=5e-2, tau=1, N=len(train_set)),
        #dict(name='msgld', lr0=5e-5, min_lr=1e-6, gamma=0.333, beta1=0.99, a=1, tau=1),
        #dict(name='asgld', lr0=5e-5, min_lr=1e-6, gamma=0.333, beta1=0.99, beta2=0.9999, eps=1e-1, a=1, tau=1),

        # Hamiltonian dynamics
        #dict(name='sghmc', lr0=1e-3, min_lr=1e-5, gamma=0.2, M=0.1, C=50),
        #dict(name='sgrmc', lr0=5e-3, min_lr=1e-6, gamma=0.333, m=1, c=100, C=100)  
    ]

    # Gather accuracy results of each optimizer
    test_errs = []
    test_accs = []
    for op in opt_params:
        err, acc = test_sgmcmc(train_set, test_set, op, epochs, batch_size, burn_in, thinning)
        test_errs.append(err)
        test_accs.append(acc)
    return np.array(test_errs), np.array(test_accs)


if __name__ == '__main__':
    # train_set = datasets.MNIST('./Data/MNIST', train=True, download=True, transform=transforms.ToTensor())
    # utils.show_some_images(train_set) => shows first 10 features
    # print(utils.calc_data_mean_std(train_set)) => mean=tensor(0.1307), std=tensor(0.3081)
    num_optimizers = len(OPTIMIZER_NAMES)
    num_experiments = 1
    epochs = 1
    batch_size = 128
    thinning = 5
    burn_in = 1
    test_errs = np.zeros(shape=(num_optimizers, epochs))
    test_accs = np.zeros(shape=(num_experiments, num_optimizers)) 

    for run in range(num_experiments):
        print('Experiment run:', run+1)
        errs, accs = fc2_test(epochs, batch_size, burn_in, thinning)
        test_errs += errs / num_experiments # calculate average test errors
        test_accs[run] = accs

    # print('Accuracy values: ', test_accs)
    # print('Accuracy mean estimate values: ', test_accs.mean(axis=0))
    # print('Accuracy std estimate values: ', test_accs.std(axis=0, ddof=1))
    # # Plot the average values
    # for err in test_errs:
    #     plt.plot([i*468 for i in range(len(err))], err)
    # plt.xlabel('Iterations')
    # plt.ylabel('Test Error')
    # plt.legend(OPTIMIZER_NAMES)
    # plt.show()

    
    
    

    