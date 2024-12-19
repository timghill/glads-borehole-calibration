import argparse
import numpy as np
import scipy
import matplotlib
matplotlib.rc('font', size=8)
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.gridspec import GridSpec
import cmocean

from sepia.SepiaPredict import SepiaEmulatorPrediction

from utils import tools, models


def eval_m_p(train_config, test_config, bh_config,
    nburn=2000, table='table.dat'):
    table = np.loadtxt(table, delimiter=' ', dtype=int)
    _,p,m = table.T
    samples = []
    for i in range(len(p)):
        mod = np.load('data/model_m{:03d}_p{:02d}.pkl'.format(m[i], p[i]), allow_pickle=True)
        samples.append(mod['samples'])

    # Load config files
    train_config = tools.import_config(train_config)
    test_config = tools.import_config(test_config)
    Y_test = np.load(test_config.Y_physical, mmap_mode='r').T

    bh_config = np.load(bh_config, allow_pickle=True)
    nodenum = bh_config['node']
    yind = nodenum*365 + np.arange(365)
    
    pcmap = cmocean.cm.amp
    mcmap = cmocean.cm.tempo
    pcolors = {
        5 : pcmap(0.2),
        10: pcmap(0.5),
        15: pcmap(0.7),
        20: pcmap(0.9),
    }
    mcolors = {
        128: mcmap(0.3),
        256: mcmap(0.5),
        512: mcmap(0.7),
        1024: mcmap(0.9),
    }

    t_test = np.loadtxt(test_config.X_standard, delimiter=',', skiprows=1)
    t_names = np.loadtxt(test_config.X_standard,
        delimiter=',', max_rows=1, dtype=str)
    # t_true = t_test[56]
    t_true = {
        128: t_test[90],
        256: t_test[90],
        512: t_test[90],
        1024:t_test[90],
    }

    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(7, 4))
    xx = np.linspace(0, 1, 51)
    for d in range(8):
        pnum = 0
        ax = axs.flat[d]
        for i in range(len(p)):
            if m[i]==512:
                smp = samples[i]['theta'].squeeze()[nburn:]
                # print(smp.shape)
                kde = scipy.stats.gaussian_kde(smp[:, d])
                ax.plot(xx, kde(xx), color=pcolors[p[i]],
                    label=p[i])
                # ax.hist(smp[:,d], bins=np.linspace(0, 1, 21),
                #     histtype='step', color=pcolors[p[i]], density=True,
                #     label=p[i])
                qii = np.quantile(smp[:,d], (0.025, 0.975))
                # ax.fill_betweenx([0, 5], [qii[0], qii[0]], [qii[1],qii[1]],
                #     color=pcolors[p[i]], alpha=0.15, edgecolor='none')
        ax.axvline(t_true[512][d], color='blue')
        ax.axhline(1, color='k', linestyle='dashed')
        ax.grid(True)
        ax.set_xlabel(t_names[d])
        ymin,ymax = ax.get_ylim()
        ax.set_ylim([0, 5])
    axs.flat[0].legend(bbox_to_anchor=(0,1,1,0.3), loc='lower left',
        ncols=4, frameon=False, title='Number of principal components')
    fig.subplots_adjust(left=0.05, right=0.95, hspace=0.3, wspace=0.3)
    fig.savefig('figures/eval_p.png', dpi=400)


    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(7, 4))
    xx = np.linspace(0, 1, 51)
    p = p[::-1]
    m = m[::-1]
    print('m:', m)
    print('p:', p)
    samples = samples[::-1]
    for d in range(8):
        ax = axs.flat[d]
        pnum = 0
        for i in range(len(p)):
            if p[i]==15:
                smp = samples[i]['theta'].squeeze()[nburn:]
                # print(smp.shape)
                kde = scipy.stats.gaussian_kde(smp[:, d])
                ax.plot(xx, kde(xx), color=mcolors[m[i]],
                    label=m[i])
                # ax.hist(smp[:,d], bins=np.linspace(0, 1, 21),
                #     histtype='step', color=mcolors[m[i]], density=True,
                #     label=m[i])
                qii = np.quantile(smp[:,d], (0.05, 0.95))
                # ax.fill_betweenx([0, 5], [qii[0], qii[0]], [qii[1],qii[1]],
                #     color=mcolors[m[i]], alpha=0.25, edgecolor='none')

        ax.axvline(t_true[m[i]][d], color='b',
            linestyle='solid')
        ax.axhline(1, color='k', linestyle='dashed')
        ax.grid(True)
        ax.set_xlabel(t_names[d])
        ymin,ymax = ax.get_ylim()
        # ax.set_ylim([0, max(2, ymax)])
        ax.set_ylim([0, 5])
    axs.flat[0].legend(bbox_to_anchor=(0,1,1,0.3), loc='lower left',
        ncols=4, frameon=False, title='Number of simulations')
    fig.subplots_adjust(left=0.05, right=0.95, hspace=0.3, wspace=0.3)
    fig.savefig('figures/eval_m.png', dpi=400)

    # fig,ax = plt.subplots(figsize=(4,4))
    fig = plt.figure(figsize=(3.5, 4))
    gs = GridSpec(2, 3, height_ratios=(100, 5), width_ratios=(10, 100, 10),
        hspace=0.15, bottom=0.075, right=0.95, left=0.1, top=0.9)
    ax = fig.add_subplot(gs[0,:])
    cax = fig.add_subplot(gs[1,1])
    for d in range(8):
        xi = 0
        ax.text(-0.05, 8-d-0.5, r'$\log$' + t_names[d], ha='right', va='center')
        for i in range(len(p)):
            if p[i]==15:
                smp = samples[i]['theta'].squeeze()[nburn:,d]
                td = t_true[m[i]][d]
                q = len(smp[smp<=td])/len(smp)
                zscore = -(td-np.mean(smp))/np.std(smp)
                vmin = -2
                vmax = 2
                znorm = (zscore-vmin)/(vmax-vmin)
                ax.fill_between([xi, xi+1], [8-d-1, 8-d-1], [8-d, 8-d],
                    color=cmocean.cm.balance(znorm))
                    # color='w')
                color = 'w' if np.abs(zscore)>=1 else 'k'
                # color = 'k'
                ax.text(xi+0.5, 8-d-1+0.67, '{:.2f}'.format(q), va='center', ha='center', color=color)
                ax.text(xi+0.5, 8-d-1+0.33, '{:.2f}'.format(zscore), va='center', ha='center', color=color)
                xi+=1
    
    ax.set_xticks(np.arange(3), ['', '', ''])
    ax.set_yticks(np.arange(8), ['', '', '', '', '', '', '' ,''])
    ax.grid(color='k')
    ax.set_xlim([0, 3])
    ax.set_ylim([0, 8])
    ax.text(1.5, 8.5, 'Number of simulations', ha='center', va='bottom')

    sm = matplotlib.cm.ScalarMappable(cmap=cmocean.cm.balance)
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    # cbar.set_ticks([0, 1])
    cbar.set_ticks((0, 0.25, 0.5, 0.75, 1))
    cbar.set_ticklabels([r'$-2\sigma$', r'$-\sigma$', r'$0$', r'$\sigma$', r'$2\sigma$'])
    cax.text(0, 1, 'Underestimate', ha='center', va='bottom')
    cax.text(1, 1, 'Overestimate', ha='center', va='bottom')

    ax.xaxis.tick_top()
    nums = [128, 256, 512]
    for i in range(3):
        ax.text(0.5+i, 8.05, nums[i], ha='center')

    fig.subplots_adjust(left=0.125, right=0.95, bottom=0.05, top=0.9)
    fig.savefig('figures/quantile_zscores.png', dpi=400)
    return

    
    fig,axs = plt.subplots(figsize=(7, 4), nrows=2, ncols=4,
        sharex=False, sharey=True)
    flierprops = dict(marker='+', markersize=1)
    medianprops = dict(color='k')
    boxprops = dict(edgecolor='none')
    dx = 0.2
    dw = 1
    xpos = np.array([np.array([1.5*dx, 0.5*dx, -0.5*dx, -1.5*dx]) + dw for dw in range(4)]).flatten()
    for d in range(8):
        ax = axs.flat[d]
        yvals = []
        for i in range(len(p)):
            betai = samples[i]['betaU'].squeeze()[nburn:,d+1, i%4]
            # print(betai.shape)
            yvals.append(betai)
            
        
        boxes = ax.boxplot(yvals, patch_artist=True,
            flierprops=flierprops, medianprops=medianprops,
            boxprops=boxprops, positions=xpos, widths=0.75*dx)
        
        for i,box in enumerate(boxes['boxes']):
            pi = p[i]
            box.set_facecolor(pcolors[pi])
        
        ax.grid(linewidth=0.5)
        ax.set_ylim([0,25])
        ax.set_title(r'$\beta($' + t_names[d] + '$)$')
        ax.set_xticks(np.arange(4), ('128', '256', '512', '1024'))
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_xlim([-0.5, 3.5])
    
    axs.flat[0].legend(handles=boxes['boxes'][:4][::-1], labels=['1', '2', '3', '4'],
        loc='upper left', ncols=2, title='Principal Component')
    
    for ax in axs[:,0]:
        ax.set_ylabel(r'$\beta$')
    left = 0.07
    fig.text(left + 0.5*(0.98-left), 0.025, 'Number of simulations', ha='center')
    fig.subplots_adjust(left=left, right=0.98, bottom=0.1, top=0.925, hspace=0.3, wspace=0.1)
    fig.savefig('figures/eval_phi.png', dpi=400)
    return

def plot_pcs(train_config, test_config, bh_config, table='table.dat'):
    table = np.loadtxt(table, delimiter=' ', dtype=int)
    _,p,m = table[::-1].T
    bh_config = np.load(bh_config, allow_pickle=True)
    node = bh_config['node']
    y_ind_sim = node*365 + np.arange(365)
    model_days = np.arange(365)
    obs_days = np.arange(365)

    # keep track of overhead time to get to MCMC sampling

    # Load config files
    train_config = tools.import_config(train_config)
    test_config = tools.import_config(test_config)

    mcmap = cmocean.cm.tempo
    mcolors = {
        128: mcmap(0.3),
        256: mcmap(0.5),
        512: mcmap(0.7),
        1024:mcmap(0.9),
    }

    mesh = np.load(train_config.mesh, allow_pickle=True)
    Kvecs = {}
    pcvar = {}

    for i in range(len(p)):
        if p[i]==5:
            train_config.p = p[i]
            train_config.m = m[i]
            # Read in the ensemble of simulations
            t_sim = np.loadtxt(train_config.X_standard, delimiter=',', skiprows=1)[:train_config.m].astype(np.float32)

            Y_sim_all = np.load(train_config.Y_physical, mmap_mode='r').T[:train_config.m, y_ind_sim]
            Y_sim = Y_sim_all[:, obs_days].astype(np.float32)

            # print(train_config.m)
            # print(t_sim.shape)
            # print(Y_sim.shape)

            # t_test = np.loadtxt(test_config.X_standard, delimiter=',', skiprows=1)
            # Y_test = np.load(test_config.Y_physical).T
            t_test = np.loadtxt(test_config.X_standard, delimiter=',', skiprows=1)[:train_config.m]
            Y_test = np.load(test_config.Y_physical).T[:train_config.m]

            # Find an appropriate candidate
            t_target = np.array([2./3., 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 3./4.])
            weights = np.array([2, 1, 1., 1., 1., 2, 1., 1])
            weighted_cost = np.sum(np.sqrt(weights**2 * (t_test - t_target)**2), axis=1)
            # sim_num = np.argmin(weighted_cost)
            sim_num = 90
            print('test sim number:', sim_num)
            
            Y_obs = Y_test[sim_num, y_ind_sim]
            t_true = t_test[sim_num]
            print(Y_obs.shape)

            data,model = models.init_model_priors(y_sim=Y_sim, y_ind_sim=None, y_obs=Y_obs[None,:], y_ind_obs=np.arange(Y_sim.shape[1]),
                t_sim=t_sim, p=train_config.p, t0=t_target)
            K = data.sim_data.K
            svals = scipy.linalg.svdvals(data.sim_data.y_std)
            pcvar[m[i]] = (svals**2)/np.sum(svals**2)
            Kvecs[m[i]] = K
    
    alphabet = ['(a)', '(b)', '(c)', '(d)']
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(7,3.5), sharex=True, sharey=True)
    for i in range(4):
        ax = axs.flat[i]
        ax.text(0.05, 0.9, alphabet[i] + r' $\rm{PC%d}$' % (i+1), transform=ax.transAxes, fontweight='bold', va='top')
        for j in range(len(m)):
            if p[j]==5:
                Ki = Kvecs[m[j]][i]
                ax.plot(Ki/np.sign(np.mean(Ki)), label=m[j],
                    color=mcolors[m[j]])

                ax.text(0.05, 0.9 - 0.025*(j+1), '{:.1f}%'.format(pcvar[m[j]][i]*100),
                    color=mcolors[m[j]], fontweight='bold', transform=ax.transAxes,
                    ha='left', va='top')

        ax.grid(linewidth=0.5)
        ax.set_xlim([0, 365])
    
    axs.flat[0].legend(loc='lower left', ncols=4, title='Number of simulations',
        bbox_to_anchor=(0,1,1,0.3), frameon=False)
    
    for ax in axs[-1]:
        ax.set_xlabel('Day of 2012')
    
    fig.text(0.02, 0.5, 'Principal component basis', ha='center', va='center', rotation=90)

    fig.subplots_adjust(left=0.08, bottom=0.15, top=0.85, right=0.98,
        hspace=0.05, wspace=0.05)
    fig.savefig('figures/eval_pc_basis.png', dpi=400)
    return


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_config')
    parser.add_argument('test_config')
    parser.add_argument('bh_config')
    parser.add_argument('--burn', required=True, type=int)
    args = parser.parse_args()

    eval_m_p(args.train_config, args.test_config, args.bh_config,
        nburn=args.burn)
    plot_pcs(args.train_config, args.test_config, args.bh_config)
