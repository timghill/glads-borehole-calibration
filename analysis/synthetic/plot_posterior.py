import os
import time
import argparse

import numpy as np
import matplotlib
matplotlib.rc('font', size=8)
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.gridspec import GridSpec
from matplotlib import cm

import cmocean

from utils import tools

def main(train_config, test_config, post_config, bh_config):
    train_config = tools.import_config(train_config)
    test_config = tools.import_config(test_config)
    post_config = tools.import_config(post_config)

    Y_prior_all = np.load(train_config.Y_physical, mmap_mode='r').T
    Y_test_all = np.load(test_config.Y_physical, mmap_mode='r').T
    Y_post_all = np.load(post_config.Y_physical, mmap_mode='r').T
    post_config.m = Y_post_all.shape[0]

    mesh = np.load(train_config.mesh, allow_pickle=True)
    mesh['connect_edge'] = tools.reorder_edges_mesh(mesh)
    mtri = Triangulation(mesh['x']/1e3, mesh['y']/1e3, mesh['elements']-1)
    
    # Load data
    bh_config = np.load(bh_config, allow_pickle=True)
    nodenum = bh_config['node']
    simnum = 90
    yind = nodenum*365 + np.arange(365)
    Y_obs_phys = Y_test_all[simnum:simnum+1, yind]
    tt = np.arange(365)

    Y_prior = Y_prior_all[:, yind]
    Y_post = Y_post_all[:256, yind]
    print('Y_post:', Y_post.shape)
    Y_post_mean = np.median(Y_post, axis=0)

    prior_qntls = np.quantile(Y_prior, np.array([0.025, 0.16, 0.85, 0.975]), axis=0)
    post_qntls = np.quantile(Y_post, np.array([0.025, 0.16, 0.84, 0.975]), axis=0)

    fig,ax = plt.subplots(figsize=(8,3))
    h1 = ax.plot(tt, Y_prior.T, color='gray', alpha=0.15, linewidth=0.15, zorder=0,
        label='Ensemble')
    f0 = ax.fill_between(tt, prior_qntls[0], prior_qntls[-1], 
        zorder=1, color='gray', alpha=0.25, edgecolor='none',
        label='Prior 95% interval')
    f1 = ax.fill_between(tt, prior_qntls[1], prior_qntls[2], 
        zorder=1, color='gray', alpha=0.4, edgecolor='none',
        label='Prior 68% interval')

    h2 = ax.plot(tt, Y_obs_phys.squeeze(), color='b', label='Synthetic observation',
        zorder=3)
    
    h3 = ax.plot(tt, Y_post_mean, color='r', label='Calibrated mean')
    f3 = ax.fill_between(tt, post_qntls[0], post_qntls[-1],
        label='Calibrated 95% interval', color='r',
        alpha=0.25, zorder=1, edgecolor='none')
    f4 = ax.fill_between(tt, post_qntls[1], post_qntls[2],
        label='Calibrated 68% interval', color='darkred', 
        alpha=0.4, zorder=1, edgecolor='none')
    
    ax.legend(handles=(h1[0], f0, f1, h3[0], f3, f4, h2[0]), 
        bbox_to_anchor=(-0.05,1,1,0.2), loc='lower left', frameon=False, ncols=4)
    ax.grid(linestyle=':')
    ax.set_ylim([0, 2.5])
    ax.set_xlim([120, 300])
    ax.set_xlabel('Day of 2012')
    ax.set_ylabel('Flotation fraction')
    fig.subplots_adjust(bottom=0.15, top=0.8, left=0.08, right=0.975)

    fig.savefig('figures/post_glads_timeseries.png', dpi=400)
    fig.savefig('figures/post_glads_timeseries.pdf')

    fig,ax = plt.subplots()
    err = Y_post_mean - Y_obs_phys
    bins = np.arange(-0.2, 0.2, 0.025)
    ax.hist(err.squeeze(), bins=bins, edgecolor='k')
    mean_err = np.mean(err)
    fig.savefig('figures/post_glads_hist.png', dpi=400)


    print('SS(err)', np.sum(err**2))
    print('SS(obs)', np.sum((Y_obs_phys - Y_obs_phys.mean())**2))
    codet = 1 - np.var(err)/np.var(Y_obs_phys)
    codet = 1 - np.sum(err**2)/np.sum((Y_obs_phys - Y_obs_phys.mean())**2)
    print('codet:', codet)
    return    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_config')
    parser.add_argument('test_config')
    parser.add_argument('post_config')
    parser.add_argument('bh_config')
    args = parser.parse_args()
    main(args.train_config, args.test_config, args.post_config, args.bh_config)


 
