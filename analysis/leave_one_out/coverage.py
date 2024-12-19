import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import cmocean

def main(table='table.dat'):
    table = np.loadtxt(table, delimiter=' ')
    t_test = np.loadtxt('../../issm/expdesign/greenland_test_standard.csv', 
        delimiter=',', skiprows=1)
    t_names = np.loadtxt('../../issm/expdesign/greenland_test_standard.csv', 
        delimiter=',', max_rows=1, dtype=str)
    ids = table[:, 1]
    njobs = len(ids)
    is_covered = np.zeros((njobs, 8))
    frac = np.zeros((njobs, 8))
    for i in range(njobs):
        model = np.load('data/model_{:03d}.pkl'.format(i+1), allow_pickle=True)
        theta = model['samples']['theta'].squeeze()[500:]
        ti = t_test[i]
        for d in range(8):
            qq = np.quantile(theta[:,d], np.array([0.025, 0.975]))
            is_covered[i,d] = np.logical_and(
                ti[d]>=qq[0], ti[d]<=qq[1])
            
            td = theta[:, d]
            q_empirical = len(td[td>=ti[d]])/len(td)
            # if q_empirical<0.5:
            frac[i, d] = 2*np.abs(0.5 - q_empirical)
            # else:
            # q_empirical = 1 - 2*np.abs(0.5
    
    coverage_frac = np.sum(is_covered, axis=0)/njobs
    print('Coverage fraction:', coverage_frac)

    check = np.sum(frac>=0.95, axis=0)
    
    print(is_covered[:5, :5])
    print(frac[:5, :5])

    print(1 - check/njobs)

    fig,ax = plt.subplots()
    ax.boxplot(frac)
    ax.set_xticks(np.arange(8)+1, t_names)
    ax.axhline(0.95, color='k', linestyle='dashed')
    # plt.show()


    fig,axs = plt.subplots(figsize=(6, 6), nrows=3, ncols=3, sharex=False, sharey=False)
    x = np.linspace(-1, 1, 101)
    err_density = np.zeros((njobs, len(x)))
    # d = 0
    [xx,yy] = np.meshgrid(x, np.arange(njobs)+1)
    for d in range(8):
        ax = axs.flat[d]
        for i in range(njobs):
            model = np.load('data/model_{:03d}.pkl'.format(i+1), allow_pickle=True)
            theta = model['samples']['theta'].squeeze()[500:]
            ti = t_test[i]
            theta_err = theta[:, d] - ti[d]
            norm_err = theta_err#/np.std(theta[:,d])
            kde = stats.gaussian_kde(norm_err)
            err_density[i] = kde(x)
        
        pc = ax.pcolormesh(xx, yy, err_density, cmap=cmocean.cm.ice_r, vmin=0, vmax=4)
        ax.axvline(0, color='k')
        ax.grid()
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xlim([-0.8, 0.8])
        ax.set_xlabel(r'$\log$' + t_names[d] + ' error')
    
    # for ax in axs[:, 0]:
    #     ax.set_ylabel('Test simulation')
    
    # for ax in axs[-1, :]:
        # ax.set_xlabel('Standard score')
    
    cax = axs.flat[-1].inset_axes((0, 0.5, 1, 0.2))
    cbar = fig.colorbar(pc, cax=cax, orientation='horizontal')
    cbar.set_label('Density')
    fig.subplots_adjust(top=0.95, right=0.95, left=0.05, bottom=0.1,
        hspace=0.5, wspace=0.1)
    # axs.flat[-1].set_visible(False)
    axs.flat[-1].spines[['left', 'right', 'bottom', 'top']].set_visible(False)
    axs.flat[-1].set_xticks([])
    axs.flat[-1].set_yticks([])
    for ax in axs[:, 1:].flat:
        ax.set_yticks([])
    
    axs[1,0].set_ylabel('Test number')

    fig.subplots_adjust(top=0.95, right=0.95, left=0.1, bottom=0.1,
        hspace=0.5, wspace=0.1)
    fig.savefig('coverage_abs_error.png', dpi=400)




    fig,axs = plt.subplots(figsize=(6, 6), nrows=3, ncols=3, sharex=False, sharey=False)
    x = np.linspace(-4, 4, 101)
    err_density = np.zeros((njobs, len(x)))
    [xx,yy] = np.meshgrid(x, np.arange(njobs)+1)
    for d in range(8):
        ax = axs.flat[d]
        for i in range(njobs):
            model = np.load('data/model_{:03d}.pkl'.format(i+1), allow_pickle=True)
            theta = model['samples']['theta'].squeeze()[500:]
            ti = t_test[i]
            theta_err = theta[:, d] - ti[d]
            norm_err = theta_err/np.std(theta[:,d])
            kde = stats.gaussian_kde(norm_err)
            err_density[i] = kde(x)
        
        pc = ax.pcolormesh(xx, yy, err_density, cmap=cmocean.cm.ice_r, vmin=0, vmax=0.5)
        ax.axvline(0, color='k')
        ax.grid()
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.set_xlim([-4, 4])
        ax.set_xlabel(r'$\log$' + t_names[d] + ' z-score')
    
    # for ax in axs[:, 0]:
    #     ax.set_ylabel('Test simulation')
    
    # for ax in axs[-1, :]:
    #     ax.set_xlabel('Standard score')

    cax = axs.flat[-1].inset_axes((0, 0.5, 1, 0.2))
    cbar = fig.colorbar(pc, cax=cax, orientation='horizontal')
    cbar.set_label('Density')
    
    # axs.flat[-1].set_visible(False)

    axs.flat[-1].spines[['left', 'right', 'bottom', 'top']].set_visible(False)
    axs.flat[-1].set_xticks([])
    axs.flat[-1].set_yticks([])

    for ax in axs[:, 1:].flat:
        ax.set_yticks([])
    
    axs[1,0].set_ylabel('Test number')

    fig.subplots_adjust(top=0.95, right=0.95, left=0.1, bottom=0.1,
        hspace=0.5, wspace=0.1)

    fig.savefig('coverage_zscore.png', dpi=400)




if __name__=='__main__':
    main()