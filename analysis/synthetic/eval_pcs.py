"""
Evaluate projections using different numbers of principal components
"""

import numpy as np
import matplotlib
fs = 8
matplotlib.rc('font', size=fs)
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import cmocean

import argparse
from utils.tools import import_config

def main(train_config, test_config, bh_config):
    train_config = import_config(train_config)
    test_config = import_config(test_config)

    Y_phys = np.load(train_config.Y_physical, mmap_mode='r').T
    mesh = np.load(train_config.mesh, allow_pickle=True)
    Y_test_raw = np.load(test_config.Y_physical, mmap_mode='r').T
    nx = mesh['numberofvertices']
    nt = int(Y_phys.shape[1]/nx)

    bh_config = np.load(bh_config, allow_pickle=True)
    nodenum = bh_config['node']
    yind = nodenum*365 + np.arange(nt)

    Y_test = Y_test_raw[:, yind]
    ntest = Y_test.shape[0]
    mvals = np.array([128, 256, 512])
    colors = cmocean.cm.tempo(np.array([0.3, 0.5, 0.7]))
    pvals = np.arange(1, 100)
    nps = len(pvals)
    Y_test_recons = np.zeros((len(mvals), nps, ntest, len(yind)))
    cvar = np.nan*np.zeros((len(mvals), nps))
    for j in range(len(mvals)):
        print(mvals[j])
        m = mvals[j]
        Y_sim = Y_phys[:m, yind]
        mu = np.mean(Y_sim, axis=0)
        sd = np.std(Y_sim, axis=0)
        sd[sd<1e-6] = 1e-6
        Y_std = (Y_sim-mu)/sd
        Z_test = (Y_test-mu)/sd
        U,S,Vh = np.linalg.svd(Y_std, full_matrices=False)
        npj = min(len(pvals), min(m, nt))
        cvar[j, :npj] = (np.cumsum(S**2)/np.sum(S**2))[:npj]
        for i in range(nps):
            pp = pvals[i]
            _U = U[:, :pp]
            _S = np.diag(S[:pp])
            _Vh = Vh[:pp]
            # Y_recons[j,i] = (mu + sd*(_U @ _S @ _Vh))[:mmin,:]

            U_test = Z_test @ _Vh.T @ np.diag(1/S[:pp])
            Y_test_recons[j,i] = mu + sd*(U_test @ _S @ _Vh)
        
    trunc_error = Y_test_recons - Y_test[None, None, :, :]
    trunc_rmse = np.sqrt(np.mean(trunc_error**2, axis=(-2,-1)))

    # Plot cumulative variance and truncation rmse
    # fig,axs = plt.subplots(nrows=2, figsize=(3,5), sharex=True)
    fig = plt.figure(figsize=(7, 3.5))
    gs = GridSpec(1, 3, width_ratios=(20, 6, 80), wspace=0.05,
        top=0.9, right=0.975, left=0.08, bottom=0.125)

    gs1 = GridSpecFromSubplotSpec(2, 1, gs[0], wspace=0.05, hspace=0.05,)
    gs2 = GridSpecFromSubplotSpec(2, 2, gs[2], wspace=0.05, hspace=0.05,)

    axs = np.array([fig.add_subplot(gs1[0]), fig.add_subplot(gs1[1])])
    for j in range(len(mvals)):
        axs[0].plot(pvals, cvar[j],
            label='m = {}'.format(mvals[j]), color=colors[j])
        axs[1].plot(pvals, trunc_rmse[j],
            label='m = {}'.format(mvals[j]), color=colors[j])
    axs[0].set_ylabel('Proportion of variance')
    axs[1].set_ylabel('RMSE')
    # for ax in axs:
    #     ax.grid(linewidth=0.5)
    #     ax.set_xlabel('Number of principal components')
    axs[1].set_xlabel('Number of PCs')
    # axs[0].legend(bbox_to_anchor=(0,1,1,0.1), ncols=len(mvals),
    #     loc='lower center', fontsize=8, frameon=False,
    #     borderpad=0., handlelength=1., handletextpad=0.5)

    alphabet = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    for i in range(2):
        axs[i].text(0.05, 0.9, alphabet[i], transform=axs[i].transAxes,
            va='bottom', ha='left', fontweight='bold')
        axs[i].grid(linewidth=0.5)
    axs[0].set_xticklabels([])
    axs[0].set_xticks((0, 50, 100))
    axs[1].set_xticks((0, 50, 100))
    axs[0].set_ylim([0.8, 1.03])
    # fig.subplots_adjust(left=0.2, bottom=0.1, right=0.95, top=0.95)
    # fig.savefig('PC_var_rmse.png', dpi=400)

    # Plot an example timeseries in basis representation
    # fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(6, 3), sharex=True, sharey=True)
    axs = np.array([[fig.add_subplot(gs2[i,j]) for j in range(2)] for i in range(2)])
    pselect = np.array([5, 10, 15, 20])
    # pselect = np.array([5, 10, 12, 15])
    simnum = 90
    # simnum = 40
    for i in range(len(pselect)):
        ax = axs.flat[i]
        ax.plot(Y_test[simnum, :], color='k', label='Exact', linewidth=2)
        for j in range(len(mvals)):
            ax.plot(Y_test_recons[j, pselect[i]-1, simnum, :], 
                label='m = {}'.format(mvals[j]), color=colors[j], linewidth=1.25)
        ax.grid(linewidth=0.5)
        ax.text(0.025, 0.9, alphabet[i+2], 
            transform=ax.transAxes, va='bottom', fontweight='bold')
        ax.text(0.975, 0.8, 'p={}\nRMSE={:.3f}'.format(pselect[i], trunc_rmse[-1, pselect[i]-1]),
            transform=ax.transAxes, va='bottom', ha='right')
        ax.set_xlim([120, 300])
        ax.set_ylim([0.5, 1.9])
    
    for ax in axs[:, 1]:
        ax.set_yticklabels([])
    
    for ax in axs[0,:]:
        ax.set_xticklabels([])
    
    for ax in axs[:, 0]:
        ax.set_ylabel('Flotation fraction')

    for ax in axs[-1,:]:
        ax.set_xlabel('Day of 2012')
    axs.flat[0].legend(bbox_to_anchor=(0, 1, 1, 0.1), ncols=(len(mvals)+1),
        loc='lower center', frameon=False)
    fig.savefig('figures/PC_timeseries.png', dpi=400)

    print('Cumulative variance for p =', pselect)
    print(cvar[:, pselect-1])
    print('Number of PCs for 0.98:')
    npcs = np.array([np.where(cvar[i,:]>=0.98)[0][0]+1 for i in range(len(mvals))])
    print(npcs)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_config')
    parser.add_argument('test_config')
    parser.add_argument('bh_config')
    args = parser.parse_args()
    main(args.train_config, args.test_config, args.bh_config)