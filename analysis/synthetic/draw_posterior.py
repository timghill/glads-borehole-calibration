"""
Draw posterior samples, evaluate and write for GlaDS evaluation
"""

import numpy as np

import matplotlib
matplotlib.rc('font', size=8)
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from utils.tools import import_config
from utils.expdesign import write_table

import argparse

def draw_posterior(config, model, nsample, nburn):
    config = import_config(config)

    model = np.load(model, allow_pickle=True)
    samples = model['samples']
    print('Read {} samples'.format(samples['theta'].shape[0]))
    theta = samples['theta'].squeeze()[nburn:]
    ntotal = theta.shape[0]
    sample_indices = np.round(np.linspace(0, ntotal-1, nsample)).astype(int)
    
    bounds = config.theta_bounds
    post_theta_std = theta[sample_indices, :]
    post_theta_log = bounds[:,0] + (bounds[:,1] - bounds[:,0])*post_theta_std
    post_theta_phys = 10**post_theta_log
    np.savetxt('post_theta_m512_std.csv', post_theta_std, delimiter=',',
        fmt='%.6e', header=','.join(config.theta_names), comments='')
    np.savetxt('post_theta_m512_phys.csv', post_theta_phys, delimiter=',',
        fmt='%.6e', header=','.join(config.theta_names), comments='')

    # 1 - QQ PLOT
    qqgrid = np.linspace(0, 1, 50)
    mcmc_qs = np.zeros((len(qqgrid), theta.shape[1]))
    post_qs = np.zeros((len(qqgrid), theta.shape[1]))
    for i in range(len(qqgrid)):
        mcmc_qs[i] = np.quantile(theta, qqgrid[i], axis=0)
        post_qs[i] = np.quantile(post_theta_std, qqgrid[i], axis=0)
    fig,ax = plt.subplots(figsize=(3,3))
    ax.plot([0,1], [0, 1], 'k')
    for i in range(theta.shape[1]):
        ax.plot(mcmc_qs[:,i], post_qs[:,i], zorder=2)
    
    ax.grid()
    ax.set_aspect('equal')
    ax.set_xlabel('MCMC')
    ax.set_ylabel('Posterior sample')
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
    # fig.savefig('figures/posterior_sample_qq.png', dpi=400)

    # 2 - CDF PLOT
    fig,ax = plt.subplots(figsize=(3, 3))
    for i in range(theta.shape[1]):
        theta_sort = np.sort(theta[:, i])
        theta_pdf = np.arange(len(theta_sort))/len(theta_sort)

        theta_post_sort = np.sort(post_theta_std[:, i])
        post_pdf = np.arange(len(post_theta_std))/len(post_theta_std)
        H = ax.plot(theta_sort, theta_pdf, alpha=0.75)
        ax.plot(theta_post_sort, post_pdf, linestyle='dashed', color=H[0].get_color())
    ax.grid(True)
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel('CDF')
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
    # fig.savefig('figures/posterior_sample_cdf.png', dpi=400)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_config')
    parser.add_argument('model')
    parser.add_argument('--number', type=int)
    parser.add_argument('--burn', type=int)
    args = parser.parse_args()
    draw_posterior(args.train_config, args.model, args.number, args.burn)