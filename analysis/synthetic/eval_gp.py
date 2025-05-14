import argparse
import numpy as np
import scipy
import matplotlib
matplotlib.rc('font', size=8)
from matplotlib import pyplot as plt
from matplotlib import patches
import cmocean

from sepia.SepiaPredict import SepiaEmulatorPrediction

from utils import tools, models


def eval_emulator(train_config, test_config, bh_config, m, p):
    train_config = tools.import_config(train_config)
    test_config = tools.import_config(test_config)

    ypred = np.load('data/pred/y_test_pred_m{:03d}_p{:02d}.npy'.format(m,p))
    ypred_mean = np.mean(ypred, axis=0)

    print(test_config.Y_physical)
    ytest = np.load(test_config.Y_physical, mmap_mode='r').T

    bh_config = np.load(bh_config, allow_pickle=True)
    print(bh_config)
    yind = bh_config['node']*365 + np.arange(365)
    ytest = ytest[:, yind]
    err = ypred_mean - ytest
    rmse = np.sqrt(np.mean(err**2, axis=-1))
    ix = np.argsort(rmse)
    nums = np.array([5, 50, 95])[::-1]
    sim_nums = ix[nums-1]

    alphabet = ['(a)', '(b)', '(c)']
    fig,axs = plt.subplots(figsize=(3.5, 4), sharex=True, nrows=3)
    for i in range(3):
        ax = axs[i]
        ax.plot(ytest[sim_nums[i]], color='k', label='GlaDS')
        ax.plot(ypred_mean[sim_nums[i]], color='r', label='Emulator mean')
        qq = np.quantile(ypred[:, sim_nums[i], :], np.array([0.025, 0.16, 0.84, 0.975]), axis=0)
        ax.fill_between(np.arange(365), qq[0], qq[-1], color='r',
            alpha=0.3, edgecolor='none', label='Emulator 95% interval')
        ax.fill_between(np.arange(365), qq[1], qq[2], color='firebrick',
            alpha=0.5, edgecolor='none', label='Emulator 68% interval')
        ax.grid(linewidth=0.5)
        ax.set_xlim([125, 300])
        ax.spines[['top', 'right']].set_visible(False)
        ax.text(0.025, 0.9, alphabet[i], transform=ax.transAxes,
            fontweight='bold', ha='left', va='top')
        
        ax.text(0.975, 0.9, 'RMSE = {:.3f}'.format(rmse[sim_nums[i]]),
            ha='right', va='top', transform=ax.transAxes)
        ylim = ax.get_ylim()
        ax.set_ylim([0, ylim[1]])
    
    axs[0].legend(bbox_to_anchor=(0,1,1,0.3), ncols=2, loc='lower left',
        frameon=False)
    
    axs[-1].set_xlabel('Day of 2012')
    axs[1].set_ylabel('Flotation fraction')
    fig.subplots_adjust(left=0.1, bottom=0.1, top=0.9, right=0.95,
        hspace=0.15)
    fig.savefig('figures/gp_timeseries.png', dpi=400)

    return


def eval_emulator_abs(train_config, test_config, bh_config, m, p):
    train_config = tools.import_config(train_config)
    test_config = tools.import_config(test_config)

    ypred = np.load('data/pred/y_test_pred_m{:03d}_p{:02d}.npy'.format(m,p))
    ypred_mean = np.mean(ypred, axis=0)

    print(test_config.Y_physical)
    ytest = np.load(test_config.Y_physical, mmap_mode='r').T

    bh_config = np.load(bh_config, allow_pickle=True)
    print(bh_config)
    yind = bh_config['node']*365 + np.arange(365)
    ytest = ytest[:, yind]
    err = ypred_mean - ytest
    rmse = np.sqrt(np.mean(err**2, axis=-1))
    ix = np.argsort(rmse)
    nums = np.array([5, 50, 95])[::-1]
    sim_nums = ix[nums-1]

    alphabet = ['(a)', '(b)', '(c)']
    fig,axs = plt.subplots(figsize=(3.5, 4), sharex=True, nrows=3)
    for i in range(3):
        ax = axs[i]
        yi = ytest[sim_nums[i]]
        # ax.plot(ytest[sim_nums[i]], color='k', label='GlaDS')
        ax.plot(ypred_mean[sim_nums[i]]-yi, color='r', label='Emulator error')
        qq = np.quantile(ypred[:, sim_nums[i], :]-yi, np.array([0.025, 0.16, 0.84, 0.975]), axis=0)
        ax.fill_between(np.arange(365), qq[0], qq[-1], color='r',
            alpha=0.3, edgecolor='none', label='Emulator error 95% interval')
        ax.fill_between(np.arange(365), qq[1], qq[2], color='firebrick',
            alpha=0.5, edgecolor='none', label='Emulator error 68% interval')
        ax.grid(linewidth=0.5)
        ax.set_xlim([125, 300])
        ax.spines[['top', 'right']].set_visible(False)
        ax.text(0.025, 0.9, alphabet[i], transform=ax.transAxes,
            fontweight='bold', ha='left', va='top')
        
        ax.text(0.975, 0.9, 'RMSE = {:.3f}'.format(rmse[sim_nums[i]]),
            ha='right', va='top', transform=ax.transAxes)
        ylim = ax.get_ylim()
        ax.set_ylim([-np.max(ylim), np.max(ylim)])
    
    axs[0].legend(bbox_to_anchor=(0,1,1,0.3), ncols=1, loc='lower left',
        frameon=False)
    
    axs[-1].set_xlabel('Day of 2012')
    axs[1].set_ylabel('Flotation fraction')
    fig.subplots_adjust(left=0.125, bottom=0.1, top=0.85, right=0.95,
        hspace=0.15)
    fig.savefig('figures/gp_timeseries_error_abs.png', dpi=400)

    return

def eval_emulator_rel(train_config, test_config, bh_config, m, p):
    train_config = tools.import_config(train_config)
    test_config = tools.import_config(test_config)

    ypred = np.load('data/pred/y_test_pred_m{:03d}_p{:02d}.npy'.format(m,p))
    ypred_mean = np.mean(ypred, axis=0)

    print(test_config.Y_physical)
    ytest = np.load(test_config.Y_physical, mmap_mode='r').T

    bh_config = np.load(bh_config, allow_pickle=True)
    print(bh_config)
    yind = bh_config['node']*365 + np.arange(365)
    ytest = ytest[:, yind]
    err = ypred_mean - ytest
    rmse = np.sqrt(np.mean(err**2, axis=-1))
    ix = np.argsort(rmse)
    nums = np.array([5, 50, 95])[::-1]
    sim_nums = ix[nums-1]

    alphabet = ['(a)', '(b)', '(c)']
    fig,axs = plt.subplots(figsize=(3.5, 4), sharex=True, nrows=3)
    for i in range(3):
        ax = axs[i]
        yi = ytest[sim_nums[i]]
        # ax.plot(ytest[sim_nums[i]], color='k', label='GlaDS')
        ax.plot((ypred_mean[sim_nums[i]]-yi)/yi, color='r', label='Emulator error')
        qq = np.quantile((ypred[:, sim_nums[i], :]-yi)/yi, np.array([0.025, 0.16, 0.84, 0.975]), axis=0)
        ax.fill_between(np.arange(365), qq[0], qq[-1], color='r',
            alpha=0.3, edgecolor='none', label='Emulator error 95% interval')
        ax.fill_between(np.arange(365), qq[1], qq[2], color='firebrick',
            alpha=0.5, edgecolor='none', label='Emulator error 68% interval')
        ax.grid(linewidth=0.5)
        ax.set_xlim([125, 300])
        ax.spines[['top', 'right']].set_visible(False)
        ax.text(0.025, 0.9, alphabet[i], transform=ax.transAxes,
            fontweight='bold', ha='left', va='top')
        
        ax.text(0.975, 0.9, 'RMSE = {:.3f}'.format(rmse[sim_nums[i]]),
            ha='right', va='top', transform=ax.transAxes)
        ylim = ax.get_ylim()
        ax.set_ylim([-np.max(ylim), np.max(ylim)])
    
    axs[0].legend(bbox_to_anchor=(0,1,1,0.3), ncols=1, loc='lower left',
        frameon=False)
    
    axs[-1].set_xlabel('Day of 2012')
    axs[1].set_ylabel('Flotation fraction')
    fig.subplots_adjust(left=0.125, bottom=0.1, top=0.85, right=0.95,
        hspace=0.15)
    fig.savefig('figures/gp_timeseries_error_rel.png', dpi=400)

    return

def eval_all_emulators(train_config, test_config, bh_config,
    table='table.dat'):
    train_config = tools.import_config(train_config)
    test_config = tools.import_config(test_config)

    bh_config = np.load(bh_config, allow_pickle=True)
    nodenum = bh_config['node']
    yind = nodenum*365 + np.arange(365)

    table = np.loadtxt(table, delimiter=' ', dtype=int)
    _,p,m = table.T

    test_preds = []
    cal_preds = []
    
    # Compute test_rmse
    Y_test = np.load(test_config.Y_physical, mmap_mode='r').T
    ntest = Y_test.shape[0]

    test_rmse = np.zeros((len(p), ntest))
    cal_rmse = np.zeros((len(p), ntest))
    for i in range(len(p)):
        test_pred = np.load('data/pred/y_test_pred_m{:03d}_p{:02d}.npy'.format(m[i], p[i]))
        test_pred_mean = np.mean(test_pred, axis=0).squeeze()
        test_err = test_pred_mean - Y_test[:, yind]
        test_rmse[i] = np.sqrt(np.mean(test_err**2, axis=1))

        cal_pred = np.load('data/pred/eta_m{:03d}_p{:02d}.npy'.format(m[i], p[i]))
        cal_mean = np.mean(cal_pred, axis=0)
        cal_err = cal_mean - Y_test[:, yind]
        cal_rmse[i] = np.sqrt(np.mean(cal_err**2, axis=1).squeeze())
    
    mcmap = cmocean.cm.tempo
    mcolors = {
        128: mcmap(0.3),
        256: mcmap(0.5),
        512: mcmap(0.7),
    }


    print('Median RMSE:')
    print(np.median(test_rmse, axis=1))

    fig,axs = plt.subplots(figsize=(3.5, 4), nrows=2, sharex=True)
    ax1,ax2 = axs
    uniqp = np.array([5, 10, 15, 20])
    dx = 1
    handles = []
    boxprops = {'edgecolor':'none'}
    medianprops = {'color':'k'}
    flierprops = {'marker':'+', 'markersize':3}
    alphabet = ['(a)', '(b)']
    for i in range(3):
        test_rmsei = test_rmse[4*i:4*(i+1)]
        cal_rmsei = cal_rmse[4*i:4*(i+1)]

        medi = np.median(test_rmsei, axis=1)
        qqi = np.quantile(test_rmsei, [0.5-0.95/2, 0.5+0.95/2], axis=1)
        mi = m[4*i]
        
        xi = uniqp + (i-1)*dx
        bp = ax1.boxplot(test_rmsei.T, positions=xi, widths=0.75,
            patch_artist=True, boxprops=boxprops, medianprops=medianprops,
            flierprops=flierprops, label=mi)
        for box in bp['boxes']:
            box.set_facecolor(mcolors[mi])
        ax2.plot(uniqp, cal_rmsei[:, 90], color=mcolors[mi], linestyle='dashed', marker='s',
            markersize=3, zorder=3)
    ax2.grid()
    yl = ax2.get_ylim()
    ax1.set_ylim([0, 0.3])
    ax2.set_ylim([0, 0.075])
    
    for i,ax in enumerate(axs):
        ax.set_xticks([5, 10, 15, 20], [5, 10, 15, 20])
        ax.spines[['right', 'top']].set_visible(False)
        ax.text(0.025, 0.95, alphabet[i], transform=ax.transAxes,
            ha='left', va='top', fontweight='bold')
    
    ax1.legend(bbox_to_anchor=(0,1,1,0.3), ncols=4, frameon=False,
        title='Number of simulations', loc='lower center')
    ax1.grid()
    ax2.set_xlabel('Number of principal components')
    ax1.set_ylabel('Test prediction RMSE')
    ax2.set_ylabel('Calibrated prediction RMSE')
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.1, top=0.9, hspace=0.05)
    fig.savefig('figures/eval_m_p.png', dpi=400)
    return


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_config')
    parser.add_argument('test_config')
    parser.add_argument('bh_config')
    parser.add_argument('--p', required=True, type=int)
    parser.add_argument('--m', required=True, type=int)
    args = parser.parse_args()

    eval_emulator(args.train_config, args.test_config, args.bh_config,
        m=args.m, p=args.p)
    eval_emulator_abs(args.train_config, args.test_config, args.bh_config,
        m=args.m, p=args.p)
    eval_emulator_rel(args.train_config, args.test_config, args.bh_config,
        m=args.m, p=args.p)
    eval_all_emulators(args.train_config, args.test_config, args.bh_config)
