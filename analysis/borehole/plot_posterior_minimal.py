"""
Plot calibrated GlaDS predictions
"""

import os
import time
import argparse

import numpy as np
import scipy.ndimage
import matplotlib
matplotlib.rc('font', size=8)
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.gridspec import GridSpec
from matplotlib import cm

import cmocean

from utils import tools

def main(train_config, post_config, bh_config):
    train_config = tools.import_config(train_config)
    post_config = tools.import_config(post_config)
    
    Y_prior_all = np.load(train_config.Y_physical, mmap_mode='r').T
    Y_post_all = np.load(post_config.Y_physical, mmap_mode='r').T
    print(post_config.Y_physical)
    post_config.m = Y_post_all.shape[0]

    mesh = np.load(train_config.mesh, allow_pickle=True)
    mesh['connect_edge'] = tools.reorder_edges_mesh(mesh)
    mtri = Triangulation(mesh['x']/1e3, mesh['y']/1e3, mesh['elements']-1)
    
    # Load data
    bh_config = np.load(bh_config, allow_pickle=True)
    nodenum = bh_config['node']
    bh_x = bh_config['x']/1e3
    bh_y = bh_config['y']/1e3
    simnum = 90
    yind = nodenum*365 + np.arange(365)
    # Y_obs_phys = Y_test_all[simnum:simnum+1, yind]
    tt = np.arange(365)

    print('bh_record:')
    
    
    bh_record = np.loadtxt(bh_config['path'], delimiter=',')
    # obs_days, Y_obs = bh_record[:199].T  # Correct for leap day missing from the model and make zero-indexed
    obs_days, Y_obs = bh_record[:199].T  # Correct for leap day missing from the model and make zero-indexed
    Y_obs = Y_obs.astype(np.float32)
    print('Raw obs_days:', obs_days)
    obs_days = (obs_days - 2).astype(int)
    print('Processed obs_days:', obs_days)
    y_ind_obs = (nodenum*365 + obs_days).astype(int)
    y_ind_sim = nodenum*365 + np.arange(365)

    Y_prior = Y_prior_all[:, yind]
    Y_post = Y_post_all[:256, yind]
    print('Y_post:', Y_post.shape)
    Y_post_mean = np.median(Y_post, axis=0)

    err = Y_post_mean[obs_days] - Y_obs
    rmse = np.sqrt(np.mean(err**2))
    print('rmse:', rmse)

    prior_qntls = np.quantile(Y_prior, np.array([0.025, 0.16, 0.84, 0.975]), axis=0)
    post_qntls = np.quantile(Y_post, np.array([0.025, 0.16, 0.84, 0.975]), axis=0)

    print('Working on plots')
    fig,ax = plt.subplots(figsize=(5,2.75))
    h1 = ax.plot(tt, Y_prior.T, color='gray', alpha=0.15, linewidth=0.15, zorder=0,
        label='Prior ensemble')
    f0 = ax.fill_between(tt, prior_qntls[0], prior_qntls[-1], 
        zorder=1, color='gray', alpha=0.25, edgecolor='none',
        label='Prior ensemble')
    # f1 = ax.fill_between(tt, prior_qntls[1], prior_qntls[2], 
    #     zorder=1, color='gray', alpha=0.4, edgecolor='none',
    #     label='Prior 68% interval')

    h2 = ax.plot(obs_days, Y_obs.squeeze(), color='b', label='Borehole data',
        zorder=3)
    
    # h3 = ax.plot(tt, Y_post_mean, color='r', label='Calibrated mean')
    # f3 = ax.fill_between(tt, post_qntls[0], post_qntls[-1],
    #     label='Calibrated ensemble', color='r',
    #     alpha=0.25, zorder=1, edgecolor='none')
    # f4 = ax.fill_between(tt, post_qntls[1], post_qntls[2],
    #     label='Calibrated 68% interval', color='darkred', 
    #     alpha=0.4, zorder=1, edgecolor='none')
    
    ax.grid(linestyle=':')
    ax.set_ylim([0, 2.5])
    ax.set_xlim([120, 300])
    ax.set_xlabel('Day of 2012')
    ax.set_ylabel('Flotation fraction')

    ax.legend(handles=(f0, h2[0]),
        bbox_to_anchor=(-0.05,1,1,0.2), loc='lower left', frameon=False, ncols=1)
    # ax2.set_ylim([0.8, 1.2])
    # ax2.set_xlim([160, 320])
    fig.subplots_adjust(bottom=0.15, top=0.8, left=0.105, right=0.975)

    # ax1.text(0.0125, 0.925, '(a)', transform=ax1.transAxes, fontweight='bold')
    # ax2.text(0.0125, 0.925, '(b)', transform=ax2.transAxes, fontweight='bold')

    fig.savefig('figures/seminar_post_timeseries_00.png', dpi=400)


    fig,ax = plt.subplots(figsize=(5,2.75))
    h1 = ax.plot(tt, Y_prior.T, color='gray', alpha=0.15, linewidth=0.15, zorder=0,
        label='Prior ensemble')
    f0 = ax.fill_between(tt, prior_qntls[0], prior_qntls[-1], 
        zorder=1, color='gray', alpha=0.25, edgecolor='none',
        label='Prior ensemble')
    # f1 = ax.fill_between(tt, prior_qntls[1], prior_qntls[2], 
    #     zorder=1, color='gray', alpha=0.4, edgecolor='none',
    #     label='Prior 68% interval')

    h2 = ax.plot(obs_days, Y_obs.squeeze(), color='b', label='Borehole data',
        zorder=3)
    
    h3 = ax.plot(tt, Y_post_mean, color='r', label='Calibrated mean')
    f3 = ax.fill_between(tt, post_qntls[0], post_qntls[-1],
        label='Calibrated ensemble', color='r',
        alpha=0.25, zorder=1, edgecolor='none')
    # f4 = ax.fill_between(tt, post_qntls[1], post_qntls[2],
    #     label='Calibrated 68% interval', color='darkred', 
    #     alpha=0.4, zorder=1, edgecolor='none')
    
    ax.grid(linestyle=':')
    ax.set_ylim([0, 2.5])
    ax.set_xlim([120, 300])
    ax.set_xlabel('Day of 2012')
    ax.set_ylabel('Flotation fraction')

    ax.legend(handles=(f0, h2[0], f3, h3[0]),
        bbox_to_anchor=(-0.05,1,1,0.2), loc='lower left', frameon=False, ncols=2)
    # ax2.set_ylim([0.8, 1.2])
    # ax2.set_xlim([160, 320])
    fig.subplots_adjust(bottom=0.15, top=0.8, left=0.105, right=0.975)

    # ax1.text(0.0125, 0.925, '(a)', transform=ax1.transAxes, fontweight='bold')
    # ax2.text(0.0125, 0.925, '(b)', transform=ax2.transAxes, fontweight='bold')

    fig.savefig('figures/seminar_post_timeseries_01.png', dpi=400)

    # NOW plot the whole thing

    fig,(ax1,ax2) = plt.subplots(figsize=(8,6), nrows=2)
    for ax in (ax1, ax2):
        h1 = ax.plot(tt, Y_prior.T, color='gray', alpha=0.15, linewidth=0.15, zorder=0,
            label='Prior ensemble')
        f0 = ax.fill_between(tt, prior_qntls[0], prior_qntls[-1], 
            zorder=1, color='gray', alpha=0.25, edgecolor='none',
            label='Prior ensemble')
        # f1 = ax.fill_between(tt, prior_qntls[1], prior_qntls[2], 
        #     zorder=1, color='gray', alpha=0.4, edgecolor='none',
        #     label='Prior 68% interval')

        h2 = ax.plot(obs_days, Y_obs.squeeze(), color='b', label='Borehole observation',
            zorder=3)
        
        h3 = ax.plot(tt, Y_post_mean, color='r', label='Calibrated mean')
        f3 = ax.fill_between(tt, post_qntls[0], post_qntls[-1],
            label='Calibrated ensemble', color='r',
            alpha=0.25, zorder=1, edgecolor='none')
        # f4 = ax.fill_between(tt, post_qntls[1], post_qntls[2],
        #     label='Calibrated 68% interval', color='darkred', 
        #     alpha=0.4, zorder=1, edgecolor='none')
        
        ax.grid(linestyle=':')
        ax.set_ylim([0, 2.5])
        ax.set_xlim([120, 364])
        ax.set_xlabel('Day of 2012')
        ax.set_ylabel('Flotation fraction')
    ax1.legend(handles=(f0, h2[0], f3, h3[0]),
        bbox_to_anchor=(-0.05,1,1,0.2), loc='lower left', frameon=False, ncols=2)
    ax2.set_ylim([0.8, 1.2])
    ax2.set_xlim([160, 320])
    fig.subplots_adjust(bottom=0.08, top=0.9, left=0.08, right=0.975, hspace=0.2)

    ax1.text(0.0125, 0.925, '(a)', transform=ax1.transAxes, fontweight='bold')
    ax2.text(0.0125, 0.925, '(b)', transform=ax2.transAxes, fontweight='bold')

    fig.savefig('figures/seminar_post_timeseries.png', dpi=400)

    return

    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_config')
    parser.add_argument('post_config')
    parser.add_argument('bh_config')
    args = parser.parse_args()
    main(args.train_config, args.post_config, args.bh_config)


 