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
from matplotlib.patches import Rectangle, Polygon
from matplotlib.collections import PatchCollection, LineCollection
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
    fig,(ax1,ax2) = plt.subplots(figsize=(8,6), nrows=2)
    for ax in (ax1, ax2):
        h1 = ax.plot(tt, Y_prior.T, color='gray', alpha=0.15, linewidth=0.15, zorder=0,
            label='Ensemble')
        f0 = ax.fill_between(tt, prior_qntls[0], prior_qntls[-1], 
            zorder=1, color='gray', alpha=0.25, edgecolor='none',
            label='Prior 95% interval')
        f1 = ax.fill_between(tt, prior_qntls[1], prior_qntls[2], 
            zorder=1, color='gray', alpha=0.4, edgecolor='none',
            label='Prior 68% interval')

        h2 = ax.plot(obs_days, Y_obs.squeeze(), color='b', label='Borehole observation',
            zorder=3)
        
        h3 = ax.plot(tt, Y_post_mean, color='r', label='Calibrated mean')
        f3 = ax.fill_between(tt, post_qntls[0], post_qntls[-1],
            label='Calibrated 95% interval', color='r',
            alpha=0.25, zorder=1, edgecolor='none')
        f4 = ax.fill_between(tt, post_qntls[1], post_qntls[2],
            label='Calibrated 68% interval', color='darkred', 
            alpha=0.4, zorder=1, edgecolor='none')
        
        ax.grid(linestyle=':')
        ax.set_ylim([0, 2.5])
        ax.set_xlim([120, 364])
        ax.set_xlabel('Day of 2012')
        ax.set_ylabel('Flotation fraction')
    ax1.legend(handles=(h1[0], f0, f1, h3[0], f3, f4, h2[0]), 
        bbox_to_anchor=(-0.05,1,1,0.2), loc='lower left', frameon=False, ncols=4)
    ax2.set_ylim([0.8, 1.2])
    ax2.set_xlim([160, 320])
    fig.subplots_adjust(bottom=0.08, top=0.9, left=0.08, right=0.975, hspace=0.2)


    rect = Rectangle(xy=(160, 0.8), width=(320-160), height=(1.2-0.8))
    pc = PatchCollection([rect], facecolor='none', edgecolor='k',
        linestyle='dashed', linewidth=0.5)
    ax1.add_collection(pc)
    ax1.text(320-2, 1.2-0.02, 'b', 
        ha='right', va='top')

    ax1.text(0.0125, 0.925, 'a', transform=ax1.transAxes, fontweight='bold')
    ax2.text(0.0125, 0.925, 'b', transform=ax2.transAxes, fontweight='bold')

    fig.savefig('figures/post_glads_timeseries.png', dpi=400)
    fig.savefig('figures/post_glads_timeseries.pdf', dpi=400)

    fig,ax = plt.subplots()
    err = Y_post_mean[obs_days] - Y_obs
    bins = np.arange(-0.2, 0.2, 0.025)
    ax.hist(err.squeeze(), bins=bins, edgecolor='k')
    mean_err = np.mean(err)
    print('Mean err:', mean_err)
    print('Median err:', np.median(err))
    print('Std err:', np.std(err))
    print('quantiles:', np.quantile(err, (0.025, 0.16, 0.86, 0.975)))
    fig.savefig('figures/post_glads_hist.png', dpi=400)

    print('SS(err)', np.sum(err**2))
    print('SS(obs)', np.sum((Y_obs - Y_obs.mean())**2))
    codet = 1 - np.var(err)/np.var(Y_obs)
    codet = 1 - np.sum(err**2)/np.sum((Y_obs - Y_obs.mean())**2)
    print('codet:', codet)

    tstep = 229
    nedges = len(mesh['edge_length'])
    post_S = np.load(post_config.Y_physical.replace('ff', 'S'), mmap_mode='r')
    post_Q = np.load(post_config.Y_physical.replace('ff', 'Q'), mmap_mode='r')
    prior_S = np.load(train_config.Y_physical.replace('ff', 'S'), mmap_mode='r')
    prior_Q = np.load(train_config.Y_physical.replace('ff', 'Q'), mmap_mode='r')

    post_S = post_S.reshape((nedges, 365, post_S.shape[-1]))
    post_Q = post_Q.reshape((nedges, 365, post_Q.shape[-1]))
    prior_S = prior_S.reshape((nedges, 365, prior_S.shape[-1]))
    prior_Q = prior_Q.reshape((nedges, 365, prior_Q.shape[-1]))
    
    print('Integrating channel area...')
    S_tot = np.sum(post_S[:, tstep, :]*mesh['edge_length'][:,None], axis=0)
    print('...done channel area')
    S_ixsort = np.argsort(S_tot)
    nums = np.round(post_config.m/100*np.array([5, 50, 95])).astype(int)-1
    ix = S_ixsort[nums]
    print('Sim nums:', ix)

    surf = np.load('../../issm/issm/data/geom/IS_surface.npy')
    zmax = 1600
    xmax = np.max(mesh['x'][surf<=zmax])/1e3
    ymax = np.max(mesh['y'][surf<=zmax])/1e3
    xmin = np.min(mesh['x'][surf<=zmax])/1e3
    ymin = np.min(mesh['y'][surf<=zmax])/1e3

    # Figure: flotation fraction and channel discharge variance for prior
    # and calibrated ensembles
    fig = plt.figure(figsize=(7,3.5))
    gs = GridSpec(3, 2, hspace=0.05, left=0.1, right=0.925, top=0.875, bottom=0.12,
        height_ratios=[8, 100, 100], width_ratios=(65, 100), wspace=0.05)
    axs = np.array([[fig.add_subplot(gs[i+1,j]) for j in range(2)] for i in range(2)])
    caxs = np.array([fig.add_subplot(gs[0,j]) for j in range(2)])
    # Qvar_prior = np.std(prior_Q[:, tstep, ::4], axis=-1)
    # Qvar_post = np.std(post_Q[:, tstep, ::4], axis=-1)
    Qvar_prior = np.abs(np.mean(prior_Q[:, tstep, ::4], axis=-1))
    Qvar_post = np.abs(np.mean(post_Q[:, tstep, ::4], axis=-1))
    # Qmin = 1
    # Qmax = 200
    var_min = 2
    var_max = 100

    # Create a custom colormap
    clist1 = cmocean.cm.dense(np.linspace(0+1e-6, 0.95, 128))
    clist2 = matplotlib.colormaps['Greys_r'](np.linspace(0.05, 0.8, 64))
    clist = np.vstack((clist1, clist2))
    w = np.ones(8)
    w = w/w.sum()
    clist = scipy.ndimage.convolve1d(clist, w, axis=0)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('name', clist)
    # cticks = np.linspace(0, 1, 6)

    # Set parameters for discrete colourmap
    cticks = np.linspace(0, var_max, 6)
    cticks[0] = var_min
    fnorm = matplotlib.colors.BoundaryNorm(cticks, 256)
    ax1 = axs[0,0]
    ax2 = axs[0,1]

    # Color and line presence/not by Q std dev
    # freq_mean = 0.5*(Qfreq_prior + Qfreq_post)
    # freq_mean = 0.5*np.abs(np.mean(prior_Q[:, tstep,:], axis=1) + np.mean(post_Q[:, tstep,:], axis=1))
    # freq_mean = np.abs(post_Q[:, tstep, 42])
    # print('mean:', freq_mean.shape)
    # freq_vals = (freq_mean, freq_mean)
    # Q_vals = (prior_Q[:,tstep,0], post_Q[:,tstep,0], post_Q[:,tstep,0])
    Q_vals = (Qvar_prior, Qvar_post)
    Y_prior_mu = np.mean(Y_prior_all[:, tstep::365], axis=0)
    Y_post_mu = np.mean(Y_post_all[:, tstep::365], axis=0)
    Y_vals = (Y_prior_mu, Y_post_mu, Y_post_mu)
    for i in range(2):
        for k in range(2):
            lscale = 1 if k==0 else 3
            ax = axs[i,k]
            pc = ax.tripcolor(mtri, Y_vals[i], cmap=cmap,
                vmin=0, vmax=1.5, alpha=0.6, rasterized=True)
            ax.plot(mesh['x'][nodenum]/1e3, mesh['y'][nodenum]/1e3, 'b^', markeredgecolor='w',
                linewidth=0.25, zorder=3)
            ax.set_aspect('equal')
            jj = np.where(Q_vals[i]>var_min)[0]
            for j in jj:
                fi = Q_vals[i][j]
                # Qi = np.abs(Q_vals[i][j])
                lw = lscale*min(1, 0.5+0.5*fnorm(fi))
                ax.plot(mesh['x'][mesh['connect_edge'][j,:]]/1e3,
                    mesh['y'][mesh['connect_edge'][j,:]]/1e3,
                    linewidth=lw, color=cmocean.cm.turbid(fnorm(fi)))

        ax1 = axs[i,0]
        ax2 = axs[i,1]
        ax1.set_xlim([xmin-1, xmax])
        ax1.set_ylim([ymin, ymax])

        ax2.set_xlim([bh_x-6, bh_x+6])
        ax2.set_ylim([bh_y-2, bh_y+2])

        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
    
    for ax in axs[:-1,:].flat:
        ax.set_xticklabels([])
    
    for ax in axs[:, 1].flat:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        ax.spines[['left', 'top']].set_visible(False)

    for ax in axs[:, 0].flat:
        ax.spines[['top', 'right']].set_visible(False)

    cb1 = fig.colorbar(pc, cax=caxs[0], orientation='horizontal', extend='max')
    sm = cm.ScalarMappable(norm=fnorm, cmap=cmocean.cm.turbid)
    cb2 = fig.colorbar(sm, cax=caxs[1], orientation='horizontal', extend='max')

    fig.text(0.01, 0.5, 'Northing (km)', rotation=90, ha='left', va='center')
    axs[-1,0].set_xlabel('Easting (km)')
    axs[-1,1].set_xlabel('Easting (km)')

    cb1.set_label('Flotation fraction')
    cb2.set_label('Mean channel discharge (m$^3$ s$^{-1}$)')
    cb2.set_ticks(cticks)
    for cax in caxs:
        cax.xaxis.tick_top()
        cax.xaxis.set_label_position('top')
    
    alphabet = ['a', 'b', 'c', 'd']
    for i,ax in enumerate(axs.flat):
        ax.text(0.025, 0.9, alphabet[i], transform=ax.transAxes,
            fontweight='bold')
    
    fig.savefig('figures/post_mean_channel_maps.png', dpi=400)
    fig.savefig('figures/post_mean_channel_maps.pdf', dpi=400)

    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_config')
    parser.add_argument('post_config')
    parser.add_argument('bh_config')
    args = parser.parse_args()
    main(args.train_config, args.post_config, args.bh_config)


 