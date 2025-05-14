import os
import time
import argparse

import numpy as np
import scipy

import matplotlib
matplotlib.rc('font', size=8)
from matplotlib import pyplot as plt

from utils import models, tools

def main(train_config, bh_config,
    nsamples=10000, nburn=2000, p=None, m=None, recompute=False):
    t_prior = np.array([2./3., 0.5, 0.5, 0.5, 0.5, 0.8, 0.5, 3./4.])

    # Borehole data
    bh_config = np.load(bh_config, allow_pickle=True)
    bh_record = np.loadtxt(bh_config['path'], delimiter=',')
    obs_days, Y_obs = bh_record[:199].T
    Y_obs = Y_obs.astype(np.float32)
    print('Raw obs_days:', obs_days)
    # Correct for leap day missing from the model and make zero-indexed
    obs_days = (obs_days - 2).astype(int)
    print('Processed obs_days:', obs_days)
    y_ind_obs = (bh_config['node']*365 + obs_days).astype(int)
    y_ind_sim = bh_config['node']*365 + np.arange(365)
    model_days = np.arange(365)

    # keep track of overhead time to get to MCMC sampling
    t0 = time.perf_counter()

    # Load config files
    train_config = tools.import_config(train_config)
    # TESTING
    if m:
        train_config.m = m
    if p:
        train_config.p = p
    p = train_config.p
    print('m:', train_config.m)
    print('p:', train_config.p)

    mesh = np.load(train_config.mesh, allow_pickle=True)

    # Read in the ensemble of simulations
    t_sim = np.loadtxt(train_config.X_standard, delimiter=',', skiprows=1)[:train_config.m].astype(np.float32)

    Y_sim_all = np.load(train_config.Y_physical, mmap_mode='r').T[:train_config.m, y_ind_sim]
    Y_sim = Y_sim_all[:, obs_days].astype(np.float32)

    # Plot ensemble and "data"
    fig,ax = plt.subplots(figsize=(8,3))
    h1 = ax.plot(model_days, Y_sim_all[:, :].T, color='gray', linewidth=0.2, alpha=0.5, label='Ensemble')
    ax.grid(linestyle=':')
    h2 = ax.plot(obs_days, Y_obs, color='blue', linewidth=1., label='GL12-2A')
    ax.legend(handles=[h1[0], h2[0]], bbox_to_anchor=(0,1,1,0.2), loc='lower left', frameon=False, ncols=2)
    ax.set_xlim([0, 364])
    ax.set_ylim([0, 2])
    ax.set_ylabel('Flotation fraction')
    ax.set_xlabel('Day of year')
    fig.subplots_adjust(bottom=0.15, top=0.8, left=0.08, right=0.975)
    fig.savefig('figures/02_ensemble_data_timeseries.png', dpi=600)

    data,model = models.init_model_priors(y_sim=Y_sim, y_ind_sim=None, y_obs=Y_obs[None,:], y_ind_obs=np.arange(Y_sim.shape[1]),
        t_sim=t_sim, p=p, t0=t_prior)
    model_file = 'data/model_m{:03d}_p{:02d}'.format(m, p)
    if not os.path.exists('data'):
        os.makedirs(data)

    if recompute:
        t1 = time.perf_counter()
        print('Time to begin MCMC sampling:', t1-t0)        
        model.do_mcmc(nsamples)
        model.save_model_info(model_file)
    else:
        model.restore_model_info(model_file)
    
    figs = models.plot_model(model, nburn, train_config, Y_sim, Y_obs, Y_ind_obs=obs_days, 
        recompute=recompute, label='GL12-2A')
    figdir = 'figures/m{:03d}_p{:02d}/'.format(train_config.m, train_config.p)
    if not os.path.exists(figdir):
        os.makedirs(figdir)
    figs[0][0].savefig('figures/03_theta_mcmc_samples.png', dpi=600)
    figs[1][0].savefig(figdir + '04_theta_posterior_distributions_m{:03d}_p{:02d}.png'.format(train_config.m, train_config.p), 
        dpi=400)
    figs[1][0].savefig(figdir + '04_theta_posterior_distributions_m{:03d}_p{:02d}.pdf'.format(train_config.m, train_config.p), 
        dpi=400)
    # figs[2][0].savefig(figdir + '05_ypred_posterior_predictions_m{:03d}_p{:02d}.png'.format(train_config.m, train_config.p), 
    #     dpi=400)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_config')
    parser.add_argument('bh_config')
    parser.add_argument('--p', required=True, type=int)
    parser.add_argument('--m', required=True, type=int)
    parser.add_argument('--sample', required=True, type=int)
    parser.add_argument('--burn', required=True, type=int)
    parser.add_argument('--recompute', action='store_true')
    args = parser.parse_args()
    main(args.train_config, args.bh_config,
        p=args.p, m=args.m, nsamples=args.sample, nburn=args.burn, 
        recompute=args.recompute)
