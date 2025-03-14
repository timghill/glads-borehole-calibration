import os
import argparse

import numpy as np
from matplotlib import pyplot as plt

import scipy

from utils import tools, models


def main(train_config, test_config, bh_config, p, m, nsamples, nburn):

    y_ind_sim = bh_config['node']*365 + np.arange(365)
    model_days = np.arange(365)
    obs_days = np.arange(365)

    # Load config files
    train_config = tools.import_config(train_config)
    test_config = tools.import_config(test_config)
    # TESTING
    if m:
        train_config.m = m
    if p:
        train_config.p = p
    p = train_config.p
    print('m:', train_config.m)
    print('p:', train_config.p)

    # mesh = np.load(train_config.mesh, allow_pickle=True)

    # Read in the ensemble of simulations
    t_sim = np.loadtxt(train_config.X_standard, delimiter=',', skiprows=1)[:train_config.m].astype(np.float32)

    Y_sim_all = np.load(train_config.Y_physical, mmap_mode='r').T[:train_config.m, y_ind_sim]
    Y_sim = Y_sim_all[:, obs_days].astype(np.float32)

    t_test = np.loadtxt(test_config.X_standard, delimiter=',', skiprows=1)
    Y_test = np.load(test_config.Y_physical, mmap_mode='r').T

    # Find an appropriate candidate
    t_target = np.array([2./3., 0.5, 0.5, 0.5, 0.5, 0.8, 0.5, 3./4.])
    weights = np.array([1, 1, 0., 0., 0., 1, 0., 1])
    weighted_cost = np.sum(np.sqrt(weights**2 * (t_test - t_target)**2), axis=1)
    sim_num = np.argmin(weighted_cost)
    print('test sim number:', sim_num)
    
    Y_obs = Y_test[sim_num, y_ind_sim]
    t_true = t_test[sim_num]

    data,model = models.init_model_priors(y_sim=Y_sim, y_ind_sim=None, y_obs=Y_obs[None,:], y_ind_obs=np.arange(Y_sim.shape[1]),
        t_sim=t_sim, p=p, t0=t_target)
    model_file = 'data/model_m{:03d}_p{:02d}'.format(m, p)
    model.restore_model_info(model_file)

    m = train_config.m
    p = train_config.p
    t_names = train_config.theta_names
    
    # TRACE PLOTS
    samples = model.get_samples()
    print(samples['theta'].shape)
    fig,ax = plt.subplots()
    for i in range(8):
        ax.plot(samples['theta'][:, i])
    
    labels = [
        'Sheet conductivity',
        'Channel conductivity',
        'Bed bump height',
        'Bed bump aspect',
        'Sheet-channel width',
        'Ice-flow coefficient',
        'Transition parameter',
        'Englacial storage',
    ]

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

    fig,axs = plt.subplots(4,2,figsize=(4,6), sharex=False, sharey=True)
    thetas = samples['theta']
    # for ax in axs.flat:
    #     ax.set_visible(False)
    for i in range(8):
        ax = axs.flat[i]
        density = scipy.stats.gaussian_kde(thetas[nburn:, i])
        xpdf = np.linspace(0, 1, 21)
        pdf = density(xpdf)
        ax.axhline(1., color='k', linestyle='dashed', label='Prior')
        ax.plot(xpdf, pdf, label='Posterior', color='red')
        if t_true is not None:
            ax.axvline(t_true[i], label='Target', color='blue')
        ax.grid(linestyle=':')
        # ax.yaxis.tick_right()
        ax.set_xlim([0, 1])
        # ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], labels=[theta_bounds[i][0], '', '','', theta_bounds[i][1]], rotation=45)
        # ax.set_xticklabels([])
        ax.set_ylim([0, 5])
        ax.set_yticks([0, 2, 4])
        # ax.set_xlabel(labels[i], labelpad=0)
        ax.text(0.5, -0.1, labels[i] + '\n' + train_config.theta_names[i], ha='center', va='top',
            transform=ax.transAxes)
    
    for ax in axs[:, 0]:
        ax.set_ylabel('Density')
    
    axs[0,0].legend(bbox_to_anchor=(0,1,1,0.3), loc='lower left', ncols=3, frameon=False)
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.925, hspace=0.5)
    
    fig.savefig('figures/thetas_posterior_minimal.png', dpi=400)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_config')
    parser.add_argument('test_config')
    parser.add_argument('--p', required=True, type=int)
    parser.add_argument('--m', required=True, type=int)
    parser.add_argument('--sample', required=True, type=int)
    parser.add_argument('--burn', required=True, type=int)
    bh_config = {   'depth':695.5, 
                    'lat':67.2042167, 
                    'lon':-49.7179333,
                    'x': -205723.84235624867,
                    'y': -2492713.333133503,
                    'node': 3611,
                    # 'node': 2508,
                    'path':'../../data/processed/GL12-2A_Pw_daily_2012.txt',
    }
    args = parser.parse_args()
    main(args.train_config, args.test_config, bh_config,
        p=args.p, m=args.m, nsamples=args.sample, nburn=args.burn)

