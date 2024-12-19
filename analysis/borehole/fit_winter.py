import os
import time
import argparse

import numpy as np
import matplotlib
matplotlib.rc('font', size=8)
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation

import scipy

from sepia.SepiaPredict import SepiaEmulatorPrediction
from sepia.SepiaModel import SepiaModel
from sepia.SepiaData import SepiaData

import cmocean

from utils import models, tools

paths = {
    'pca': 'data/pca',
}

def main(train_config, bh_config,
    nsamples=10000, nburn=2000, m=None, recompute=False):
    t_prior = np.array([2./3., 0.5, 0.5, 0.5, 0.5, 0.8, 0.5, 3./4.])

    bh_config = np.load(bh_config, allow_pickle=True)
    bh_x= bh_config['x']
    bh_y = bh_config['y']
    nodenum = bh_config['node']

    bh_record = np.loadtxt(bh_config['path'], delimiter=',')
    # obs_days, Y_obs = bh_record[:199].T  # Correct for leap day missing from the model and make zero-indexed
    obs_days, Y_obs = bh_record[:199].T  # Correct for leap day missing from the model and make zero-indexed
    Y_obs = Y_obs.astype(np.float32)
    obs_days = (obs_days - 2).astype(int)
    y_ind_obs = (nodenum*365 + obs_days).astype(int)
    y_ind_sim = nodenum*365 + np.arange(365)
    model_days = np.arange(365)

    # keep track of overhead time to get to MCMC sampling
    t0 = time.perf_counter()

    # Load config files
    train_config = tools.import_config(train_config)
    # TESTING
    if m:
        train_config.m = m

    mesh = np.load(train_config.mesh, allow_pickle=True)

    # Read in the ensemble of simulations
    t_sim = np.loadtxt(train_config.X_standard, delimiter=',', skiprows=1)[:train_config.m].astype(np.float32)
    t_names = train_config.theta_names

    Y_sim_all = np.load(train_config.Y_physical, mmap_mode='r').T[:train_config.m, y_ind_sim]
    Y_sim = Y_sim_all[:, obs_days].astype(np.float32)

    # Take an average of the last 30 days of the record
    print(Y_obs.shape)
    print(Y_sim.shape)
    Y_obs = np.array([[Y_obs[-30:].mean()]])
    Y_sim = np.mean(Y_sim[:, -30:], axis=1)[:, None]
    print(Y_obs.shape)
    print(Y_sim.shape)

    data = SepiaData(t_sim=t_sim, y_sim=Y_sim, y_obs=Y_obs)
    data.standardize_y()
    data.transform_xt(t_notrans=np.arange(t_sim.shape[1]))
    model = SepiaModel(data)
    model.params.theta.val = np.array([t_prior])

    model.do_mcmc(nsamples)
    model_file = 'data/winter_m{:03d}'.format(m)
    if not os.path.exists('data'):
        os.makedirs(data)
    model.save_model_info(model_file)
    # model.restore_model_info(model_file)

    preds = SepiaEmulatorPrediction(model=model, samples=model.get_samples())
    ypred = preds.get_y()

    fig,ax = plt.subplots(figsize=(3, 3))
    ax.boxplot(ypred.squeeze(), patch_artist=True,
        boxprops=dict(edgecolor='k', facecolor='r'),
        medianprops=dict(color='k'),
        flierprops=dict(marker='+', markersize=5),
    )
    ax.axhline(Y_obs.squeeze(), color='b')
    ax.set_ylabel('Flotation fraction')
    ax.set_xticklabels([])
    fig.subplots_adjust(left=0.2, bottom=0.05, right=0.95, top=0.95)

    figdir = 'figures/winter/m_{:03d}/'.format(m)
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    fig.savefig(figdir + 'winter_boxplot.png'.format(m), dpi=400)

    
    # POSTERIOR DISTRIBUTIONS
    theta_bounds = np.array([
                        ['$10^{-3}$', '$10^{-1}$'],                # Sheet conductivity
                        ['$10^{-1}$', '$10^{0}$'],
                        [r'$5\times 10^{-2}$', '1.0'],    # Bed bump height
                        ['$10^1$', '$10^2$'],                 # Bed bump aspect ratio
                        ['$10^0$', '$10^2$'],                 # Channel-sheet width
                        ['$10^{-24}$', '$10^{-22}$'],             # Rheology parameter (C&P Table 3.3 p 73)
                        ['1/500', '1/5000'],
                        ['$10^{-4}$', '$10^{-3}$'],
    ])
    samples = model.get_samples(nburn=0)
    fig,axs = plt.subplots(8,8,figsize=(8,8), sharex=False, sharey=False)
    thetas = samples['theta']
    print("thetas:", thetas.shape)
    for ax in axs.flat:
        ax.set_visible(False)
    for row in range(8):
        for col in range(row):
            ax = axs[row,col]
            ax.set_visible(True)
            # ax.scatter(thetas[50:, col], thetas[50:, row], 3, samples['logPost'][50:], cmap=cmocean.cm.rain, vmin=520, vmax=560)
            
            kdei = scipy.stats.gaussian_kde(np.array([thetas[nburn:, col], thetas[nburn:, row]]))
            x1 = np.linspace(0, 1, 21)
            x2 = np.linspace(0, 1, 21)
            xx1,xx2 = np.meshgrid(x1,x2)
            xq = np.array([xx1.flatten(), xx2.flatten()])
            zz = kdei(xq).reshape(xx1.shape)
            ax.pcolormesh(xx1, xx2, zz, cmap=cmocean.cm.amp, vmin=0)
            # ax.plot(t_true[col], t_true[row], 'b+', markeredgewidth=2.5, markeredgecolor='w', markersize=10)
            # ax.plot(t_true[col], t_true[row], 'b+', markeredgewidth=1.5, markeredgecolor='b', markersize=10)

            # zqq = np.quantile(zz.flatten(), 0.05)
            zordered = np.sort(zz.flatten())
            zsum = np.cumsum(zordered)
            zindex = np.argmin( (zsum - 0.05*zsum[-1])**2)
            zqq = zordered[zindex]
            # print('zindex:', zindex)
            # print('zsum:', zsum)
            # ax.contour(xx1, xx2, zz, levels=[zqq,], colors='k', linestyles='dashed', linewidths=0.5)
            
            if col==0:
                ax.text(-0.4, 0.5, t_names[row], rotation=90, va='center', ha='right')
            if row==7:
                ax.text(0.5, -0.4, t_names[col], rotation=0, va='top', ha='center')
            # ax.grid(linestyle=':')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.])
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.])
            
            r_test = scipy.stats.pearsonr(thetas[nburn:, col], thetas[nburn:, row])
            r = r_test.statistic
            # pvalue = r_test.pvalue
            # if pvalue
            axT = axs[col, row]
            axT.set_visible(True)
            textcolor = 'k' if np.abs(r)<0.8 else 'w'
            axT.text(0.5, 0.5, '{:.2f}'.format(r), transform=axT.transAxes, 
                ha='center', va='center', color=textcolor, fontweight='bold')
            axT.set_xticks([])
            axT.set_yticks([])
            axT.set_facecolor(cmocean.cm.balance(0.5*(1 +r)))
            
            # ax.set_xticks([0, 1], theta_bounds[col])
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1], 
                [theta_bounds[row][0], '', '', '', theta_bounds[row][1]])
            if col>0:
                ax.set_yticklabels([])
            if row<7:
                ax.set_xticklabels([])

        ax = axs[row,row]
        ax.set_visible(True)
        density = scipy.stats.gaussian_kde(thetas[nburn:, row])
        xpdf = np.linspace(0, 1, 21)
        pdf = density(xpdf)
        ax.axhline(1., color='k', linestyle='dashed', label='Prior')
        ax.plot(xpdf, pdf, label='Posterior', color='red')
        # ax.axvline(t_true[row], label='Target', color='blue')
        ax.grid(linestyle=':')
        ax.yaxis.tick_right()
        ax.set_xlim([0, 1])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels([])
        ax.set_ylim([0, 5])
        ax.set_yticks([0, 2, 4])

    axs[0,0].legend(bbox_to_anchor=(0., 1.0, 0.5, 1), loc='lower left', frameon=False, ncols=3)
    axs[0,0].text(-0.4, 0.5, t_names[0], rotation=90, va='center', ha='right',
        transform=axs[0,0].transAxes)
    # axs[-1,-1].set_xlabel(t_names[-1])
    axs[-1,-1].text(0.5, -0.4, t_names[-1], rotation=0, va='top', ha='center',
        transform=axs[-1,-1].transAxes)
    for i,ax in enumerate(axs[-1, :]):
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], labels=[theta_bounds[i][0], '', '','', theta_bounds[i][1]], rotation=45)
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.075, top=0.95, wspace=0.3, hspace=0.2)
    
    fig.savefig(figdir+'04_theta_posterior_distribution_winter.png', dpi=400)
    return


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_config')
    parser.add_argument('bh_config')
    parser.add_argument('--m', required=True, type=int)
    parser.add_argument('--sample', required=True, type=int)
    parser.add_argument('--burn', required=True, type=int)
    parser.add_argument('--recompute', action='store_true')
    args = parser.parse_args()
    main(args.train_config, args.bh_config,
        m=args.m, nsamples=args.sample, nburn=args.burn, 
        recompute=args.recompute)
