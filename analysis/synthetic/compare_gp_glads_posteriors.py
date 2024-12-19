import numpy as np
import matplotlib
matplotlib.rc('font', size=8)
from matplotlib import pyplot as plt

def main():
    ypred = np.load('data/pred/eta_m512_p15.npy').squeeze()

    y_post = np.load('../../issm/issm/post_synthetic/greenland_ff.npy', mmap_mode='r').T
    nodenum = 3611
    yind = 365*nodenum + np.arange(365)
    y_post = y_post[:, yind]
    print('y_post.shape:', y_post.shape)

    y_test = np.load('../../issm/issm/test/greenland_ff.npy', mmap_mode='r').T
    simnum = 90
    y_test = y_test[simnum, yind]
    print('y_test.shape:', y_test.shape)

    qq = np.array([0.025, 0.25, 0.75, 0.975])
    q_gp = np.quantile(ypred, qq, axis=0)
    q_glads = np.quantile(y_post, qq, axis=0)

    mu_gp = np.median(ypred, axis=0)
    mu_glads = np.median(y_post, axis=0)
    tt = np.arange(365)
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.plot(tt, y_test, color='b', label='Synthetic data')
    
    ax.plot(tt, mu_gp, color='k', label='Emulator')
    ax.fill_between(tt, q_gp[0], q_gp[-1], color='gray',
        alpha=0.4, edgecolor='none')
    ax.plot(tt, mu_glads, color='r', label='GlaDS')
    ax.fill_between(tt, q_glads[0], q_glads[-1], color='r',
        alpha=0.4, edgecolor='none')
    ax.legend(bbox_to_anchor=(0,0.7,1,0.3), loc='upper right', 
        ncols=3, frameon=True)
    ax.set_ylabel('Flotation fraction')
    ax.set_xlabel('Day of 2012')
    ax.set_xlim([120, 300])
    ax.grid(linewidth=0.5)
    plt.subplots_adjust(left=0.12, bottom=0.15, right=0.98, top=0.975)
    
    fig.savefig('figures/compare_gp_glads_posteriors.png', dpi=400)

if __name__=='__main__':
    main()