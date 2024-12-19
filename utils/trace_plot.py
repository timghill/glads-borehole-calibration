"""
Simple trace plots
"""

import numpy as np

import matplotlib
matplotlib.rc('font', size=8)
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from utils.tools import import_config

import argparse

def trace_plot(config, model, nburn=0, figpattern='trace_{}.png'):
    config = import_config(config)

    model = np.load(model, allow_pickle=True)
    samples = model['samples']
    theta = samples['theta'].squeeze()[nburn:]
    beta = samples['betaU'].squeeze()[nburn:]
    lam_obs = samples['lamOs'].squeeze()[nburn:]
    lam_sim = samples['lamWOs'].squeeze()[nburn:]
    lam_gp  = samples['lamUz'].squeeze()[nburn:]
    lam_nug = samples['lamWs'].squeeze()[nburn:]

    print('theta:', theta.shape)
    print('beta:', beta.shape)
    print('lam_sim:', lam_sim.shape)
    print('lam_nug:', lam_nug.shape)
    print('lam_obs:', lam_obs.shape)
    print('lam_gp:', lam_gp.shape)
    nsamples, ndim = theta.shape
    if lam_gp.ndim>1:
        npc = lam_gp.shape[1]
    else:
        npc = 0

    fig = plt.figure(figsize=(7,5))
    gs = GridSpec(4, 2, left=0.07, right=0.975, bottom=0.1, top=0.95,
        hspace=0.3, wspace=0.15)
    axs = np.array([[fig.add_subplot(gs[i,j]) for j in range(2)] for i in range(4)])

    for i in range(ndim):
        ax = axs.flat[i]
        ax.plot(theta[:,i])
        ax.set_title(config.theta_names[i])
    
    for ax in axs[:-1].flat:
        ax.set_xticklabels([])
    for ax in axs.flat:
        ax.grid(True)
    
    for ax in axs[-1]:
        ax.set_xlabel('Samples')

    # plt.show()
    fig.savefig('figures/'+figpattern.format('theta'), dpi=400)

    fig = plt.figure(figsize=(7,7))
    gs = GridSpec(6, 2, left=0.07, right=0.975, bottom=0.1, top=0.95,
        hspace=0.3, wspace=0.15)
    axs = np.array([[fig.add_subplot(gs[i,j]) for j in range(2)] for i in range(6)])

    for i in range(ndim):
        ax = axs.flat[i]
        ax.plot(beta[:,i+1], color='tab:blue', linewidth=0.5, alpha=0.5)
        ax.set_title(r'$\beta$(' + config.theta_names[i] + ')')
    
    for ax in axs[:-1].flat:
        ax.set_xticklabels([])
    
    for ax in axs[-1]:
        ax.set_xlabel('Samples')

    axs.flat[8].plot(lam_obs)
    axs.flat[9].plot(lam_sim)
    axs.flat[10].plot(lam_gp, color='tab:blue', linewidth=0.5, alpha=0.5)
    axs.flat[11].plot(lam_nug, color='tab:blue', linewidth=0.5, alpha=0.5)
    for ax in axs[:-1].flat:
        ax.set_xticklabels([])

    for ax in axs.flat:
        ax.grid()

    axs.flat[8].set_title(r'$\lambda_{y}$')
    axs.flat[9].set_title(r'$\lambda_{\eta}$')
    axs.flat[10].set_title(r'$\lambda_{w}$')
    axs.flat[11].set_title(r'$\lambda_{\rm{n}}$')

    fig.savefig('figures/'+figpattern.format('phi'), dpi=400)

    return

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_config')
    parser.add_argument('model')
    parser.add_argument('--burn', type=int, required=True)
    parser.add_argument('--save', default='trace_{}.png')
    args = parser.parse_args()
    trace_plot(args.train_config, args.model, nburn=args.burn,
        figpattern=args.save)