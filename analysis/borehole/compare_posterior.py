"""
Compare calibrated model uncertainty between synthetic and borehole
calibration experiments.

Computes the ensemble spread (95%) flotation fraction and histograms
of channel volume.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.rc('font', size=7)
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib import cm

import scipy.stats

import cmocean

from utils import tools

def main(train_config, synth_config, bh_config, bh_data):
    train_config = tools.import_config(train_config)
    synth_config = tools.import_config(synth_config)
    bh_config = tools.import_config(bh_config)

    Y_prior_all = np.load(train_config.Y_physical, mmap_mode='r').T
    Y_synth_all = np.load(synth_config.Y_physical, mmap_mode='r').T
    Y_bh_all = np.load(bh_config.Y_physical, mmap_mode='r').T

    S_prior = np.load(train_config.Y_physical.replace('ff', 'S'), mmap_mode='r').T
    S_synth = np.load(synth_config.Y_physical.replace('ff', 'S'), mmap_mode='r').T
    S_bh = np.load(bh_config.Y_physical.replace('ff', 'S'), mmap_mode='r').T

    #Q_prior = np.load(train_config.Y_physical.replace('ff', 'Q'), mmap_mode='r').T
    #Q_synth = np.load(synth_config.Y_physical.replace('ff', 'Q'), mmap_mode='r').T
    #Q_bh = np.load(bh_config.Y_physical.replace('ff', 'Q'), mmap_mode='r').T

    mesh = np.load(train_config.mesh, allow_pickle=True)
    mtri = Triangulation(mesh['x']/1e3, mesh['y']/1e3, mesh['elements']-1)

    bh_data = np.load(bh_data, allow_pickle=True)
    nodenum = bh_data['node']
    bh_x = bh_data['x']/1e3
    bh_y = bh_data['y']/1e3
    simnum = 90
    tt = np.arange(365)

    bh_record = np.loadtxt(bh_data['path'], delimiter=',')
    obs_days, Y_obs = bh_record[:199].T
    Y_obs = Y_obs.astype(np.float32)
    # Correct for leap day missing from the model and make zero-indexed
    obs_days = (obs_days - 2).astype(int)
    y_ind_obs = (nodenum*365 + obs_days).astype(int)
    y_ind_sim = nodenum*365 + np.arange(365)

    Y_prior = Y_prior_all[:, y_ind_sim]
    Y_synth = Y_synth_all[:256, y_ind_sim]
    Y_bh = Y_bh_all[:256, y_ind_sim]

    qq = np.array([0.025, 0.25, 0.5, 0.75, 0.975])
    nv = mesh['numberofvertices']
    nt = 365
    yprior = Y_prior_all.T.reshape((nv,nt,Y_prior_all.shape[0]))
    ysynth = Y_synth_all.T.reshape((nv,nt,Y_synth_all.shape[0]))
    ybh = Y_bh_all.T.reshape((nv,nt,Y_bh_all.shape[0]))

    # map quantiles
    unc_ind = np.arange(140, 300)
    map_qq_prior = np.quantile(yprior[:,unc_ind,:], qq, axis=-1)
    map_qq_prior = np.mean(map_qq_prior, axis=-1)
    # map_qq_synth = np.quantile(Y_synth_all[:, 229::365], qq, axis=0)
    
    map_qq_synth = np.quantile(ysynth[:,unc_ind,:], qq, axis=-1)
    map_qq_synth = np.mean(map_qq_synth, axis=-1)
    
    map_qq_bh = np.quantile(ybh[:,unc_ind,:], qq, axis=-1)
    map_qq_bh = np.mean(map_qq_bh, axis=-1)

    # FIGURE
    fig = plt.figure(figsize=(6, 4))
    gs = GridSpec(1, 2, width_ratios=(125, 100),
        wspace=0.25, left=0.1, bottom=0.1, right=0.975, top=0.95,
        hspace=0.,
    )

    gs_maps = GridSpecFromSubplotSpec(3, 2, gs[0], hspace=-0.1,
        width_ratios=(5, 100), wspace=0.05)
    axs_maps = np.array([fig.add_subplot(gs_maps[i,1]) for i in range(3)])
    cax = fig.add_subplot(gs_maps[1,0])

    bndry = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    cticks = bndry.astype(str)
    cticks[:] = ''
    cticks[0] = '$10^{-2}$'
    cticks[9] = '$10^{-1}$'
    cticks[18] = '$10^{0}$'

    logbndry = np.log10(bndry)
    cnorm = matplotlib.colors.Normalize(vmin=-2, vmax=0)
    delta = np.log10(1e-4 + np.abs(map_qq_prior[-1]-map_qq_prior[0]))
    mpbl = axs_maps[0].tripcolor(mtri, delta,
        cmap=cmocean.cm.rain, norm=cnorm, rasterized=True)
    axs_maps[1].tripcolor(mtri, np.log10(map_qq_synth[-1]-map_qq_synth[0]),
        cmap=cmocean.cm.rain, norm=cnorm, rasterized=True)
    axs_maps[2].tripcolor(mtri, np.log10(map_qq_bh[-1]-map_qq_bh[0]),
        cmap=cmocean.cm.rain, norm=cnorm, rasterized=True)
    for ax in axs_maps[1:]:
        ax.plot(bh_x, bh_y, 'b^', markeredgecolor='w', linewidth=0.5,
            label='GL12-2A')
    axs_maps[-1].legend(bbox_to_anchor=(0,-0.4,1,0.5),
        loc='upper center', frameon=False)

    cbar = fig.colorbar(mpbl, cax=cax, orientation='vertical', spacing='proportional')
    cbar.set_label('Flotation fraction ensemble spread')
    cax.yaxis.tick_left()
    cax.yaxis.set_label_position('left')
    cbar.set_ticks(logbndry)
    cbar.set_ticklabels(cticks)

    # Compute ratio of uncertainty
    uncert_ratio_synth = (map_qq_prior[-1]-map_qq_prior[0])/(map_qq_synth[-1]-map_qq_synth[0])
    uncert_ratio_bh = (map_qq_prior[-1]-map_qq_prior[0])/(map_qq_bh[-1]-map_qq_bh[0])
    print('Uncertainty ratios (median, sd)')
    print('Synthetic:', np.nanmedian(uncert_ratio_synth), np.nanstd(uncert_ratio_synth))
    print('Borehole :', np.nanmedian(uncert_ratio_bh), np.nanstd(uncert_ratio_bh))

    gs_hist = GridSpecFromSubplotSpec(2, 1, gs[1], hspace=0.15)
    axs_hist = np.array([fig.add_subplot(gs_hist[i]) for i in range(2)])

    tstep = 229
    print(S_prior.shape)
    S_tot_prior = np.sum(S_prior[:, tstep::365]*mesh['edge_length'], axis=-1)
    S_tot_synth = np.sum(S_synth[:, tstep::365]*mesh['edge_length'], axis=-1)
    S_tot_bh = np.sum(S_bh[:, tstep::365]*mesh['edge_length'], axis=-1)

    priorBins = np.arange(0, 5e7, 0.4e7)
    postBins = np.arange(0, 5e7, 0.2e7)
    priorCounts,priorBins = np.histogram(S_tot_prior, bins=priorBins, density=True)
    synthCounts, synthBins = np.histogram(S_tot_synth, bins=postBins, density=True)
    bhCounts, bhBins = np.histogram(S_tot_bh, bins=postBins, density=True)
    axs_hist[0].bar(priorBins[:-1], priorCounts, width=priorBins[1]-priorBins[0], 
        label='Prior', color='gray', 
        alpha=2./3., edgecolor='#222222', align='edge', linewidth=0.5)
    axs_hist[0].bar(synthBins[:-1], synthCounts, width=postBins[1]-postBins[0],
        label='Calibrated', color='r', 
        alpha=2./3., edgecolor='#222222', align='edge', linewidth=0.5)

    axs_hist[0].axvline(12714700.234662913, color='b', label='True')

    axs_hist[1].bar(priorBins[:-1], priorCounts, width=priorBins[1]-priorBins[0], 
        label='Prior', color='gray', 
        alpha=2./3., edgecolor='#222222', align='edge', linewidth=0.5)
    axs_hist[1].bar(bhBins[:-1], bhCounts, width=postBins[1]-postBins[0], 
        label='Calibrated', color='r', 
        alpha=2./3., edgecolor='#222222', align='edge', linewidth=0.5)
    
    # Styling
    surf = np.load('../../issm/issm/data/geom/IS_surface.npy')
    zmax = 1850
    xmax = np.max(mesh['x'][surf<=zmax])/1e3
    ymax = np.max(mesh['y'][surf<=zmax])/1e3
    xmin = np.min(mesh['x'][surf<=zmax])/1e3
    ymin = np.min(mesh['y'][surf<=zmax])/1e3
    for ax in axs_maps:
        ax.set_aspect('equal')
        ax.set_xlim((xmin,xmax))
        ax.set_ylim((ymin,ymax))
        ax.spines[['left', 'right', 'bottom', 'top']].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    axs_maps[0].text(0.025, 0.9, '$\\bf{a}$ Prior',
        transform=axs_maps[0].transAxes, ha='left', va='bottom')
    axs_maps[1].text(0.025, 0.9, '$\\bf{b}$ Synthetic',
        transform=axs_maps[1].transAxes, ha='left', va='bottom')
    axs_maps[2].text(0.025, 0.9, '$\\bf{c}$ Borehole',
        transform=axs_maps[2].transAxes, ha='left', va='bottom')

    for ax in axs_hist:
        ax.set_ylim([0, 2e-7])
        ax.grid(linewidth=0.3, zorder=0)
        ax.spines[['right', 'top']].set_visible(False)
        
    axs_hist[0].set_xticklabels([])
    axs_hist[0].set_ylabel('Density', labelpad=2)
    axs_hist[1].set_ylabel('Density', labelpad=2)
    axs_hist[1].set_xlabel('Total channel volume (m$^3$)')
    axs_hist[0].text(0.025, 0.925, 'd', transform=axs_hist[0].transAxes,
        fontweight='bold')
    axs_hist[1].text(0.025, 0.925, 'e', transform=axs_hist[1].transAxes,
        fontweight='bold')
    axs_hist[0].legend(loc='upper right', frameon=True, title='Synthetic')
    axs_hist[1].legend(loc='upper right', title='Borehole')
    
    fig.savefig('figures/posterior_comparison.png', dpi=400)
    fig.savefig('figures/posterior_comparison.pdf', dpi=400)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_config')
    parser.add_argument('post_synth_config')
    parser.add_argument('post_borehole_config')
    parser.add_argument('bh_config')
    args = parser.parse_args()
    main(args.train_config, args.post_synth_config, 
        args.post_borehole_config, args.bh_config)
