import argparse
import numpy as np
import matplotlib
matplotlib.rc('font', size=7)
from matplotlib import pyplot as plt
import scipy
import cmocean

from utils.tools import import_config

# def main(models, config, burn=500):
def main(config, models, burn=500):
    md_all = np.load(models[0], allow_pickle=True)
    md_summer = np.load(models[1], allow_pickle=True)
    md_winter = np.load(models[2], allow_pickle=True)
    models = (md_all, md_summer, md_winter)

    config = import_config(config)

    colors = ['gray', cmocean.cm.balance(0.75), cmocean.cm.balance(0.25)]

    # theta_bounds = np.array([
    #                     ['$10^{-3}$', '$10^{-1}$'],                # Sheet conductivity
    #                     ['$10^{-1}$', '$10^{0}$'],
    #                     [r'$5\times 10^{-2}$', '1.0'],    # Bed bump height
    #                     ['$10^1$', '$10^2$'],                 # Bed bump aspect ratio
    #                     ['$10^0$', '$10^2$'],                 # Channel-sheet width
    #                     ['$10^{-24}$', '$10^{-22}$'],             # Rheology parameter (C&P Table 3.3 p 73)
    #                     ['1/500', '1/5000'],
    #                     ['$10^{-4}$', '$10^{-3}$'],
    # ])
    theta_bounds = np.array([
                        ['-3', '-1'],                # Sheet conductivity
                        ['$-1$', '0'],
                        [r'$\log(0.05)$', '0'],    # Bed bump height
                        ['1', '2'],                 # Bed bump aspect ratio
                        ['0', '2'],                 # Channel-sheet width
                        ['-24', '-22'],             # Rheology parameter (C&P Table 3.3 p 73)
                        [r'$\log(1/5000)$', r'$\log(1/500)$'],
                        ['-4', '-3'],
    ])

    theta_names = [  
                    r'$k_{\rm{s}}$',# (${\rm{Pa}}\,{\rm{s}^{-1}})$',
                    r'$k_{\rm{c}}$',# ($\rm{m^{3/2}}\,s^{-1}$)', 
                    r'$h_{\rm{b}}$',# ($\rm{m}$)',
                    r'$r_{\rm{b}}$',# (-)',
                    r'$l_{\rm{c}}$',# ($\rm{m}$)',
                    r'$A$',# (${\rm{s}}^{-1}\,{\rm{Pa}}^{-3}$)',
                    r'$\omega$',# (-)',
                    r'$e_{\rm{v}}$',# (-)',
    ]




    fig, axs = plt.subplots(4, 2, sharey=False,
        figsize=(3.5, 5))
    bins = 10**np.linspace(0, 1, 51)
    for i in range(8):
        ax = axs.flat[i]
        logmin = config.theta_bounds[i][0]
        logmax = config.theta_bounds[i][1]
        xi_log = np.linspace(logmin, logmax, 51)
        xi_phys = 10**xi_log
        labels = ['All', 'Summer', 'Winter']
        ax.axhline(1/np.abs(logmax-logmin), color='k', linestyle='dashed', label='Prior')
        for j in range(3):
            # print(xi_log[::20])
            # print(xi_phys[::20])

            # xi = np.linspace(config.theta_bounds[i][0], config.theta_bounds[i][1], 101)
            # print(xi[::10])
            smp = models[j]['samples']
            # theta = 10**smp['theta']
            theta_log = logmin + (logmax-logmin)*smp['theta'][:,:,i].squeeze()[burn:]
            print(theta_log.shape)
            # theta = 10**(xi[0] + (xi[-1]-xi[0])*smp['theta'][:, :, i].squeeze())
            # theta_phys = 10**theta_log
            # print(theta_phys.min(), theta_phys.max())
            # print('theta:', theta.shape)
            # print(theta[:10])
            # ax.hist(theta[:, :, i], bins=bins)
            kde = scipy.stats.gaussian_kde(theta_log)
            vals = kde(xi_log)
            ax.plot(xi_log, vals, color=colors[j], label=labels[j])
            ax.fill_between(xi_log, 0*vals, vals, color=colors[j], alpha=0.1)
            # ax.set_ylim([0, 4/np.abs(logmax - logmin)])
            # ax.set_xscale('log')
            # ax.set_xlim((10**logmin, 10**logmax))
        ax.set_xticks(np.linspace(logmin, logmax, 5), [theta_bounds[i][0], '', '', '', theta_bounds[i][1]])
        # ax.set_xticks(np.linspace(logmin, logmax, 5), [logmin, '', '', '', logmax])
        ax.grid(linestyle=':')
        ax.set_xlabel(r'$\log$' + theta_names[i], labelpad=-6)
        ymax = ax.get_ylim()[1]
        if ymax<(2/np.abs(logmax-logmin)):
            ymax = 2/np.abs(logmax-logmin)
        ax.set_ylim([0, ymax])
        # ax.set_ylim([0, 4])

    
    for ax in axs[:, 0]:
        ax.set_ylabel('Density')
    
    # for ax in axs[:-1,:].flat:
    #     ax.set_xticklabels([])
    
    
    axs[0,0].legend(bbox_to_anchor=(0,1,1,1), loc='lower left', ncols=4,
        frameon=False)
    fig.subplots_adjust(bottom=0.05, top=0.95, right=0.98, left=0.15, 
        hspace=0.3, wspace=0.22)

    fig.savefig('figures/separate_calibration.png', dpi=400)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_config')
    parser.add_argument('models', nargs=3)
    parser.add_argument('--burn', type=int, required=True)
    args = parser.parse_args()
    main(args.train_config, args.models, burn=args.burn)
