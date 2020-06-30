import torch
import torch.nn as nn
from math import log, pi
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


def tensor_gaussian_log_likelihood(x, mean, logvar, clip=True):
    if clip:
        logvar = torch.clamp(logvar, min=-4, max=3)
    a = log(2*pi)
    b = logvar
    c = (x - mean)**2 / torch.exp(logvar)
    return -0.5 * torch.sum(a + b + c, dim=-1)


def tensor_gaussian_log_likelihood_per_dim(x, mean, logvar, clip=True):
    if clip:
        logvar = torch.clamp(logvar, min=-4, max=3)
    a = log(2*pi)
    b = logvar
    c = (x - mean)**2 / torch.exp(logvar)
    return -0.5 * (a + b + c)


def gaussian_log_likelihood(x, mean, logvar, clip=True):
    if clip:
        logvar = torch.clamp(logvar, min=-4, max=3)
    a = log(2*pi)
    b = logvar
    c = (x - mean)**2 / torch.exp(logvar)
    return -0.5 * torch.sum(a + b + c)


def L2_loss(x, mean):
    loss_fn = nn.MSELoss()
    return loss_fn(x, mean)



def bernoulli_log_likelihood(x, p, clip=True, eps=1e-6):
    if clip:
        p = torch.clamp(p, min=eps, max=1 - eps)
    return torch.sum((x * torch.log(p)) + ((1 - x) * torch.log(1 - p)))


def tensor_kl_diagnormal_stdnormal(mean, logvar):
    a = mean**2
    b = torch.exp(logvar)
    c = -1
    d = -logvar
    return 0.5 * torch.sum(a + b + c + d, dim=-1)

def kl_diagnormal_stdnormal(mean, logvar):
    a = mean**2
    b = torch.exp(logvar)
    c = -1
    d = -logvar
    return 0.5 * torch.sum(a + b + c + d)


def tensor_kl_diagnormal_diagnormal_dim(q_mean, q_logvar, p_mean, p_logvar):
    a = p_logvar
    b = - 1
    c = - q_logvar
    d = ((q_mean - p_mean)**2 + torch.exp(q_logvar)) / torch.exp(p_logvar)

    return 0.5 * (a + b + c + d)


def tensor_kl_diagnormal_diagnormal(q_mean, q_logvar, p_mean, p_logvar):
    a = p_logvar
    b = - 1
    c = - q_logvar
    d = ((q_mean - p_mean)**2 + torch.exp(q_logvar)) / torch.exp(p_logvar)

    return 0.5 * torch.sum(a + b + c + d, dim=-1)


def kl_diagnormal_diagnormal(q_mean, q_logvar, p_mean, p_logvar):
    # Ensure correct shapes since no numpy broadcasting yet
    p_mean = p_mean.expand_as(q_mean)
    p_logvar = p_logvar.expand_as(q_logvar)

    a = p_logvar
    b = - 1
    c = - q_logvar
    d = ((q_mean - p_mean)**2 + torch.exp(q_logvar)) / torch.exp(p_logvar)
    return 0.5 * torch.sum(a + b + c + d)




"""
embedding: The (N, X) floating point data where X is either 2 or 3 depending on
    if drawing a 2D or 3D plot.
labels: The (N,) integer labels refering a name in the distributions parameter.
    There should therefore be L unique elements in this array
distributions: The (L,) string list of the names to use in the legend
"""
def scatter_contexts(embedding, labels, distributions, savepath=None, no_label=False,
    use_map_dict=False):
    fig = plt.figure()

    colors = sns.hls_palette(n_colors=len(distributions), l=0.6, s=.9, h=0.7)

    map_dict = {
    'quadrant 1' : 'North-East',
    'quadrant 2' : 'North-West',
    'quadrant 3' : 'South-West',
    'quadrant 4' : 'South-East'
    }

    # 2D Plot
    if embedding.shape[-1] == 2:
        ax = fig.add_subplot(111)

        embedding = np.array(embedding).reshape(-1, 2)
        n = len(embedding)
        labels = labels[:n]
        ix = [np.where(labels == label)
                for i, label in enumerate(distributions)]
        for label, i in enumerate(ix):
            ax.scatter(embedding[i][:, 0], embedding[i][:, 1],
                        label=distributions[label].title(),
                        color=colors[label])

    # 3D Plot
    elif embedding.shape[-1] == 3:
        ax = fig.add_subplot(111, projection='3d')
        embedding = np.array(embedding).reshape(-1, 3)

        n = len(embedding)
        labels = labels[:n]
        ix = [np.where(labels == label)
              for i, label in enumerate(distributions)]

        for label, i in enumerate(ix):
            if use_map_dict:
                ax.scatter(embedding[i][:, 0], embedding[i][:, 1], embedding[i][:, 2],
                           label=map_dict[distributions[label]].title(),
                           color=colors[label])
            else:
                ax.scatter(embedding[i][:, 0], embedding[i][:, 1], embedding[i][:, 2],
                           label=distributions[label].title(),
                           color=colors[label])

    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False,
                    right=False, left=False, labelleft=False)

    legend = plt.legend(loc='best', facecolor='#FAFAFA', framealpha=1, shadow=True)
    frame = legend.get_frame()
    frame.set_linewidth(0)

    if no_label:
        ax.get_legend().remove()

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    if savepath is not None:
        print('saved fig to %s' % savepath)
        plt.savefig(savepath)


def contexts_by_moment(contexts, moments, savepath=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    contexts = np.array(contexts).reshape(-1, 3)
    cax = ax.scatter(contexts[:, 0], contexts[:, 1], contexts[:, 2],
                     c=moments[:len(contexts)])
    fig.colorbar(cax)

    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False,
                    right=False, left=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath)
