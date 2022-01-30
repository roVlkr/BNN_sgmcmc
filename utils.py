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

def conv_shape(in_channels: int, out_channels: int, kernel_size: int, stride: int=1, padding: int=0):
    return (in_channels, out_channels, kernel_size, stride, padding)

def pool_shape(kernel_size: int, stride: int=None, padding: int=0):
    stride = kernel_size if stride == None else stride # performs a // kernel_size operation
    return (kernel_size, stride, padding)

def conv_to_linear_shapes(feature_size: np.ndarray, conv_layout: list):
    if len(conv_layout) == 0:
        return [feature_size.prod()]
    linear_shapes = [feature_size.prod() * conv_layout[0][0]]
    for conv_layer, pool_layer in zip(conv_layout[::2], conv_layout[1::2]):
        _, out_channels, kernel_size, stride, padding = conv_layer
        # Conv
        feature_size[0] = (feature_size[0] + 2*padding - kernel_size) // stride + 1
        feature_size[1] = (feature_size[1] + 2*padding - kernel_size) // stride + 1
        # Pool
        kernel_size, stride, padding = pool_layer
        feature_size[0] = (feature_size[0] + 2*padding - kernel_size) // stride + 1
        feature_size[1] = (feature_size[1] + 2*padding - kernel_size) // stride + 1
        linear_shapes.append(out_channels * feature_size.prod())
    return linear_shapes
        