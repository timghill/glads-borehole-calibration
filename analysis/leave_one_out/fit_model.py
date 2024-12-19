import os
import time
import argparse

import numpy as np
import scipy

import matplotlib
matplotlib.rc('font', size=8)
from matplotlib import pyplot as plt

from sepia.SepiaParam import SepiaParam

from utils import models, tools

def main(train_config, test_config, bh_config, sim_num,
    nsamples=10000, nburn=2000, p=None, m=None, recompute=False):
    y_ind_sim = bh_config['node']*365 + np.arange(365)
    model_days = np.arange(365)
    obs_days = np.arange(365)

    # keep track of overhead time to get to MCMC sampling
    t0 = time.perf_counter()

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

    mesh = np.load(train_config.mesh, allow_pickle=True)

    # Read in the ensemble of simulations
    t_sim = np.loadtxt(train_config.X_standard, delimiter=',', skiprows=1)[:train_config.m].astype(np.float32)

    Y_sim_all = np.load(train_config.Y_physical, mmap_mode='r').T[:train_config.m, y_ind_sim]
    Y_sim = Y_sim_all[:, obs_days].astype(np.float32)

    t_test = np.loadtxt(test_config.X_standard, delimiter=',', skiprows=1)
    Y_test = np.load(test_config.Y_physical).T

    print('test sim number:', sim_num)
    
    Y_obs = Y_test[sim_num, y_ind_sim]
    t_true = t_test[sim_num]
    print('t_true:', t_true)


    data,model = models.init_model_priors(y_sim=Y_sim, y_ind_sim=None, y_obs=Y_obs[None,:], y_ind_obs=np.arange(Y_sim.shape[1]),
        t_sim=t_sim, p=p, t0=t_true.copy())
    model_file = 'data/model_{:03d}'.format(sim_num+1)
    if not os.path.exists('data'):
        os.makedirs('data')


    model.params.lamOs  = SepiaParam(val=1, name='lamOs',
        val_shape=(1, 1), dist='Gamma', params=[5, 5],
        bounds=[0.01, np.inf], mcmcStepParam=1.0, mcmcStepType='Uniform')
    model.params.mcmcList = [model.params.theta, model.params.betaU,
        model.params.lamUz, model.params.lamWs, model.params.lamWOs,
        model.params.lamOs]

    if recompute:
        t1 = time.perf_counter()
        print('Time to begin MCMC sampling:', t1-t0)        
        model.do_mcmc(nsamples)
        model.save_model_info(model_file)
    else:
        model.restore_model_info(model_file)
    
    figs = models.plot_model(model, nburn, train_config, Y_sim, Y_obs, Y_ind_obs=obs_days, recompute=recompute,
        label='Synthetic observation', t_true=t_true)
    figdir = 'figures/{:03d}/'.format(sim_num+1)
    if not os.path.exists(figdir):
        os.makedirs(figdir)
    figs[1][0].savefig(figdir + '04_theta_posterior_distributions_m{:03d}_p{:02d}.png'.format(train_config.m, train_config.p), 
        dpi=300)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_config')
    parser.add_argument('test_config')
    parser.add_argument('bh_config')
    parser.add_argument('jobid', type=int)
    parser.add_argument('--p', required=True, type=int)
    parser.add_argument('--m', required=True, type=int)
    parser.add_argument('--sample', required=True, type=int)
    parser.add_argument('--burn', required=True, type=int)
    parser.add_argument('--recompute', action='store_true')
    args = parser.parse_args()
    bh_config = np.load(args.bh_config, allow_pickle=True)
    main(args.train_config, args.test_config, bh_config, sim_num=args.jobid-1,
        p=args.p, m=args.m, nsamples=args.sample, nburn=args.burn, 
        recompute=args.recompute)
