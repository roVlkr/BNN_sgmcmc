import os
from xml.etree.ElementInclude import DEFAULT_MAX_INCLUSION_DEPTH
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from train_procedure import Training, Evaluation
from fcmodel import FCModel
from priors import *
from optimizer import *
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

def show_some_images(dataset, n=10):
    image_loader = DataLoader(dataset, batch_size=n)
    images, _ = next(iter(image_loader))
    figure = make_grid(images[:12], nrow=12)
    plt.figure(figsize=(10,4))
    plt.imshow(np.transpose(figure.numpy(), (1, 2, 0)))
    plt.show()

def calc_data_mean_std(dataset):    
    loader = DataLoader(dataset, batch_size=1000)
    num_batches = len(loader)
    mean = 0
    for (x, _) in loader:
        mean += x.mean()
    mean /= num_batches
    std = 0 # biased
    for (x, _) in loader:
        std += (x - mean).square_().mean()
    std /= num_batches
    return mean, torch.sqrt(std)

def plot_distributions(wpriors, bpriors, wposts, bposts):
    num_rows = len(wpriors) # for each layer one row
    fig, axes = plt.subplots(nrows=num_rows, ncols=2)    
    fig.suptitle('Priors vs. Posteriors')
    for i, (wprior, wpost) in enumerate(zip(wpriors, wposts)):
        axes[i, 0].set_title(f'Layer {i+1}: Weights')
        sns.kdeplot(wprior, ax=axes[i, 0], fill=True)
        sns.kdeplot(wpost, ax=axes[i, 0], fill=True)
        axes[i, 0].legend(['Prior', 'Posterior'])
    for i, (bprior, bpost) in enumerate(zip(bpriors, bposts)):
        axes[i, 1].set_title(f'Layer {i+1}: Biases')
        sns.kdeplot(bprior, ax=axes[i, 1], fill=True)
        sns.kdeplot(bpost, ax=axes[i, 1], fill=True)
        axes[i, 1].legend(['Prior', 'Posterior'])
    plt.tight_layout()
    plt.show()

def test_sgmcmc(train_set, test_set, opt_params, epochs, batch_size, thinning):
    print(f'Training configuration: Epochs {epochs}, Batch size {batch_size}, Thinning {thinning}')
    print(f'Optimizer: {opt_params}')
    
    network_layout = [784, 400, 100, 10]
    # ~ Xavier init
    wpriors=[Laplace(0, math.pow(n, -0.5)) for n in network_layout[:-1]]
    bpriors=[Laplace(0, math.pow(n, 0)) for n in network_layout[:-1]]
    model = FCModel(network_layout, wpriors, bpriors).cuda()
    weights, biases = model.weights_biases() # record initial weights
    wpriors = [w.detach().cpu().view(-1) for w in weights]
    bpriors = [b.detach().cpu().view(-1) for b in biases]

    training = Training(train_set, test_set, model, batch_size, opt_params)
    param_records, losses, test_err = training.start(epochs, thinning)
    weights, biases = model.weights_biases() # record initial weights
    wposts = [w.detach().cpu().view(-1) for w in weights]
    bposts = [b.detach().cpu().view(-1) for b in biases]

    evaluation = Evaluation(test_set, model, param_records, batch_size=1000)
    #acc = evaluation.eval_all()
    #print(f'Bayes acc: {acc}')

    plot_distributions(wpriors, bpriors, wposts, bposts)
    return losses, test_err

if __name__ == '__main__':
    # Show some images
    # image_set = datasets.MNIST('./Data/MNIST', train=True, download=True, transform=transforms.ToTensor())
    # show_some_images(image_set)
    # Define Data
    # print(calc_data_mean_std(train_set)) => tensor(0.1307), tensor(0.3081)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.1307, std=0.3081)
    ])
    train_set = datasets.MNIST('./Data/MNIST', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./Data/MNIST', train=False, download=True, transform=transform)

    # Define optimizers
    opt_params = [
        #dict(name='sgld', lr0=1e-5, min_lr=5e-7, gamma=0.333, tau=1),
        dict(name='psgld', lr0=5e-7, min_lr=1e-8, gamma=0.333, alpha=0.999, eps=1e-2, tau=1, N=len(train_set)),
        #dict(name='msgld', lr0=1e-5, min_lr=5e-7, gamma=0.333, beta1=0.99, a=1, tau=1),
        #dict(name='asgld', lr0=1e-5, min_lr=5e-7, gamma=0.333, beta1=0.99, beta2=0.9999, eps=1e-2, a=1, tau=1)

        # 
        #dict(name='sghmc', lr0=1e-3, min_lr=1e-6, gamma=0.2, M=1, C=100)    
    ]

    # Gather accuracy results of each optimizer
    test_errs = []
    train_losses = []
    epochs = 10
    batch_size = 128
    thinning = 5
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

    
    
    

    