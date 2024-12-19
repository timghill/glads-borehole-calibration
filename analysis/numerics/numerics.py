import argparse
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.gridspec import GridSpec
import cmocean

from utils.tools import import_config

def numerics(coarse_config, fine_config, bh_config):
    coarse_config = import_config(coarse_config)
    fine_config = import_config(fine_config)
    bh_config = np.load(bh_config, allow_pickle=True)

    mesh = np.load(fine_config.mesh, allow_pickle=True)
    nx = mesh['numberofvertices']
    nt = 365

    Y_coarse = np.load(coarse_config.Y_physical, mmap_mode='r')
    Y_fine = np.load(fine_config.Y_physical, mmap_mode='r')

    m = min(Y_coarse.shape[1], Y_fine.shape[1])
    
    t_sim = np.loadtxt(fine_config.X_standard,
        delimiter=',', skiprows=1)[:m]
    t_names = np.loadtxt(fine_config.X_standard,
        delimiter=',', max_rows=1, dtype=str)

    dY = Y_fine - Y_coarse[:, :m]
    sim_rmse = np.sqrt(np.mean(dY**2, axis=0))

    dY = dY.reshape((nx, nt, m))
    map_rmse = np.sqrt(np.mean(dY**2, axis=(1,2)))

    # dY[dY>=0.1] = np.nan
    time_rmse = np.sqrt(np.nanmean(dY[bh_config['node'],:,:]**2, axis=(1)))

    mtri = Triangulation(mesh['x']/1e3, mesh['y']/1e3, mesh['elements']-1)
    # fig,(ax,ax2) = plt.subplots(figsize=(6,6), nrows=2)
    fig = plt.figure(figsize=(6,6.5))
    gs = GridSpec(2, 2, width_ratios=(100, 5), height_ratios=(100, 100),
        left=0.135, right=0.9, bottom=0.1, top=0.95, hspace=0.3, wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1,0])
    cax = fig.add_subplot(gs[0,1])
    tpc = ax.tripcolor(mtri, map_rmse, vmin=0, vmax=1e-1, cmap=cmocean.cm.speed)

    surf = np.load('../../issm/issm/data/geom/IS_surface.npy')
    ela = 1850
    xmin = np.min(mesh['x'][surf<=ela]) - 5e3
    xmax = np.max(mesh['x'][surf<=ela])
    ymin = np.min(mesh['y'][surf<=ela])
    ymax = np.max(mesh['y'][surf<=ela])
    ax.set_xlim((xmin/1e3, xmax/1e3))
    ax.set_ylim((ymin/1e3, ymax/1e3))
    cbar = fig.colorbar(tpc, extend='max', cax=cax)
    cbar.set_label('Flotation fraction RMSE')
    ax.set_aspect('equal')
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel('Easting (km)')
    ax.set_ylabel('Northing (km)')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # fig.subplots_adjust(bottom=0.1, left=0.1, right=0.95, top=0.95)
    # fig.savefig('sim_discrepancy_map.png', dpi=400)

    # fig,ax2 = plt.subplots(figsize=(8,4))
    ax2.plot(np.arange(365), dY[bh_config['node'], :, 0], 
        color='gray', alpha=0.4, linewidth=0.5, label='Ensemble')
    ax2.plot(np.arange(365), dY[bh_config['node'], :, 1:], 
        color='gray', alpha=0.4, linewidth=0.5)
    ax2.grid()

    dY_ts = dY[bh_config['node'], :, :]

    qq = np.quantile(dY_ts, (0.025, 0.5-0.68/2, 0.5+0.68/2, 0.975), axis=1)
    q2 = np.quantile(dY_ts, 0.975, axis=1)

    ax2.fill_between(np.arange(365), qq[0], qq[-1], zorder=5, alpha=0.3, 
        color='red', edgecolor='none', label='95% interval')
    ax2.fill_between(np.arange(365), qq[1], qq[2], zorder=5, alpha=0.5,
        color='firebrick', edgecolor='none', label='68% interval')
    ax2.set_xlabel('Day of 2012')
    ax2.set_ylabel(r'$\Delta$Flotation fraction')
    ax2.set_ylim([-0.3, 0.3])
    ax2.set_xlim([0, 365])
    ax2.spines[['right', 'top']].set_visible(False)
    # ax2.plot(np.arange(365), dY[bh_config['node'], :, 90], color='k', label='Synthetic data')
    ax2.legend(bbox_to_anchor=(0,1,1,0.3), loc='lower center', ncols=4, frameon=False)
    ax.text(0.025, 0.925, '(a)', fontweight='bold', transform=ax.transAxes)
    ax2.text(0.025, 0.925, '(b)', fontweight='bold', transform=ax2.transAxes)
    fig.subplots_adjust(bottom=0.125, left=0.1, right=0.95, top=0.9)
    fig.savefig('sim_discrepancy_timeseries.png', dpi=400)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('coarse_config')
    parser.add_argument('fine_config')
    parser.add_argument('bh_config')
    args = parser.parse_args()
    numerics(args.coarse_config, args.fine_config, args.bh_config)
