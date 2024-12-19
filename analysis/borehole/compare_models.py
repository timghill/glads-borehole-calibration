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


def eval_m_p(train_config, bh_config,
    nburn=2000, table='table.dat'):
    table = np.loadtxt(table, delimiter=' ', dtype=int)
    _,p,m = table.T
    samples = []
    for i in range(len(p)):
        mod = np.load('data/model_m{:03d}_p{:02d}.pkl'.format(m[i], p[i]), allow_pickle=True)
        samples.append(mod['samples'])
        
    # Load config files
    train_config = tools.import_config(train_config)

    bh_config = np.load(bh_config, allow_pickle=True)
    nodenum = bh_config['node']
    yind = nodenum*365 + np.arange(365)
    
    pcmap = cmocean.cm.amp
    mcmap = cmocean.cm.tempo
    pcolors = {
        5 : pcmap(0.2),
        10: pcmap(0.5),
        12: pcmap(0.7),
        15: pcmap(0.9),
    }
    mcolors = {
        128: mcmap(0.3),
        256: mcmap(0.5),
        512: mcmap(0.7),
    }

    t_names = np.loadtxt(train_config.X_standard,
        delimiter=',', max_rows=1, dtype=str)

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
        # ax.axvline(t_true[256][d], color='blue')
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
            if p[i]==12:
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

            # ax.axvline(t_true[m[i]][d], color=mcolors[m[i]],
            #     linestyle='dashed')
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
    
    fig,axs = plt.subplots(figsize=(7, 4), nrows=2, ncols=4,
        sharex=False, sharey=True)
    flierprops = dict(marker='+', markersize=1)
    medianprops = dict(color='k')
    boxprops = dict(edgecolor='none')
    dx = 0.2
    dw = 1
    xpos = np.array([np.array([1.5*dx, 0.5*dx, -0.5*dx, -1.5*dx]) + dw for dw in range(3)]).flatten()
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

def plot_pcs(train_config, bh_config, table='table.dat'):
    table = np.loadtxt(table, delimiter=' ', dtype=int)
    _,p,m = table[::-1].T
    bh_config = np.load(bh_config, allow_pickle=True)
    node = bh_config['node']
    
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

    # Load config files
    train_config = tools.import_config(train_config)

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

            data,model = models.init_model_priors(y_sim=Y_sim, y_ind_sim=None, y_obs=Y_obs[None,:], y_ind_obs=np.arange(Y_sim.shape[1]),
                t_sim=t_sim, p=train_config.p, t0=np.ones((1,8)))
            K = data.sim_data.K
            svals = scipy.linalg.svdvals(data.sim_data.y_std)
            pcvar[m[i]] = (svals**2)/np.sum(svals**2)
            Kvecs[m[i]] = K

    alphabet = ['(a)', '(b)', '(c)', '(d)']
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(7,3.5), sharex=True, sharey=True)
    for i in range(4):
        ax = axs.flat[i]
        ax.text(0.025, 0.985, alphabet[i] + r' $\rm{PC%d}$' % (i+1), transform=ax.transAxes, fontweight='bold', va='top')
        for j in range(len(m)):
            if p[j]==5:
                Ki = Kvecs[m[j]][i]
                ax.plot(obs_days, Ki/np.sign(np.mean(Ki)), label=m[j],
                    color=mcolors[m[j]])

                ax.text(0.95, 0.9 - 0.025*j, '{:.1f}%'.format(pcvar[m[j]][i]*100),
                    color=mcolors[m[j]], fontweight='bold', transform=ax.transAxes,
                    ha='right', va='top')

        ax.grid(linewidth=0.5)
        ax.set_xlim([obs_days[0], obs_days[-1]])
        ax.set_ylim([-0.5, 1.2])
    
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
    parser.add_argument('bh_config')
    parser.add_argument('--burn', required=True, type=int)
    args = parser.parse_args()

    eval_m_p(args.train_config, args.bh_config,
        nburn=args.burn)
    plot_pcs(args.train_config, args.bh_config)
