import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

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