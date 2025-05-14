import os
import time
import numpy as np
# from utils import svd
import scipy

import matplotlib
matplotlib.rc('font', size=8)
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import cmocean

from sepia.SepiaData import SepiaData
from sepia.SepiaModel import SepiaModel
from sepia.SepiaParam import SepiaParam
from sepia.SepiaPredict import SepiaPrediction, SepiaEmulatorPrediction

def compute_K_basis(Y_sim):
    # Compute PC basis
    # U_, S_, Vh_ = svd.randomized_svd(Y_sim, 25, k=0, q=1)
    U,S_,Vh = np.linalg.svd(Y_sim, full_matrices=False)
    S = np.diag(S_)

    # Cumulative variance for curiosity
    cvar = np.cumsum(S_**2)/np.sum(S_**2)
    print(cvar[:15])

    K = S @ Vh / np.sqrt(Y_sim.shape[0])
    return U,S,Vh,K

def init_model_priors(y_sim, y_ind_sim, y_obs, y_ind_obs, t_sim, p, t0):
    data = SepiaData(t_sim=t_sim, y_sim=y_sim, y_ind_sim=np.arange(y_sim.shape[1]),
        y_obs=y_obs, y_ind_obs=np.arange(y_sim.shape[1]))
    data.transform_xt(t_notrans=np.arange(t_sim.shape[1]))
    mu_y = np.mean(y_sim, axis=0).astype(np.float32)
    sd_y = np.std(y_sim, axis=0).astype(np.float32)
    data.standardize_y(y_mean=mu_y, y_sd=sd_y)
    data.create_K_basis(p)
    model = SepiaModel(data)

    svals = scipy.linalg.svdvals(data.sim_data.y_std)
    pcvar = np.cumsum(svals**2)/np.sum(svals**2)
    print('PC expl var:', pcvar[:25])

    # Priors
    # Residual variance of PC truncated basis representation
    pc_prec = 1/(1 - pcvar[p-1])

    def gamma_optimizer(a, mu, width, level=0.99):
        b = (a-1)/mu
        rv = scipy.stats.gamma(a=a, scale=1/b)
        q1 = rv.ppf((1-level)/2)
        q2 = rv.ppf((1+level)/2)
        return width - (q2-q1)
    
    m = y_sim.shape[0]
    if m<=256:
        level = 0.95
    elif m<=512:
        level = 0.99
    else:
        level = 0.999

    optfun = lambda a: gamma_optimizer(a, pc_prec, pc_prec/2, level=level)
    gamma_a = scipy.optimize.fsolve(optfun, x0=50)[0]
    gamma_b = (gamma_a-1)/pc_prec
    print('PC precision:', pc_prec)
    print('PC precision prior parameters:', gamma_a, gamma_b)

    # Assume small uncertainty in observations to construct prior, this is sampled in MCMC chain
    os_prec = 1
    print('obs precision:', os_prec)
    model.params.lamOs  = SepiaParam(val=os_prec, name='lamOs',
        val_shape=(1, 1), dist='Gamma', params=[5, 5],
        bounds=[0.01, np.inf], mcmcStepParam=1, mcmcStepType='Uniform')
    model.params.lamWOs = SepiaParam(val=pc_prec, name='lamWOs', 
        val_shape=(1, 1), dist='Gamma', params=[gamma_a, gamma_b], 
        bounds=[pc_prec/2, np.inf], mcmcStepParam=5, mcmcStepType='Uniform')
    model.params.theta.val = t0[None,:].copy()
    model.params.mcmcList = [model.params.theta, model.params.betaU,
        model.params.lamUz, model.params.lamWs, model.params.lamWOs,
        model.params.lamOs]

    return data,model


def plot_model(model, nburn, train_config, Y_sim, Y_obs, Y_ind_obs, 
    label, t_true=None, recompute=False):
    m = train_config.m
    p = train_config.p
    t_names = train_config.theta_names
    
    # TRACE PLOTS
    samples = model.get_samples()
    print(samples['theta'].shape)
    fig,ax = plt.subplots()
    for i in range(8):
        ax.plot(samples['theta'][:, i])

    figaxs = []
    figaxs.append((fig,ax))

    # POSTERIOR DISTRIBUTIONS
    theta_bounds = np.array([
                        ['$10^{-3}$', '$10^{-1}$'],                # Sheet conductivity
                        ['$10^{-1}$', '$10^{0}$'],
                        [r'$5\times 10^{-2}$', '1.0'],    # Bed bump height
                        ['$10^1$', '$10^2$'],                 # Bed bump aspect ratio
                        ['$10^0$', '$10^2$'],                 # Channel-sheet width
                        ['$10^{-24}$', '$10^{-22}$'],             # Rheology parameter (C&P Table 3.3 p 73)
                        ['1/500', '1/5000'],
                        ['$10^{-4}$', '$10^{-3}$'],
    ])

    fig,axs = plt.subplots(8,8,figsize=(8,8), sharex=False, sharey=False)
    thetas = samples['theta']
    for ax in axs.flat:
        ax.set_visible(False)
    for row in range(8):
        for col in range(row):
            ax = axs[row,col]
            ax.set_visible(True)
            # ax.scatter(thetas[50:, col], thetas[50:, row], 3, samples['logPost'][50:], cmap=cmocean.cm.rain, vmin=520, vmax=560)
            
            kdei = scipy.stats.gaussian_kde(np.array([thetas[nburn:, col], thetas[nburn:, row]]))
            x1 = np.linspace(0, 1, 21)
            x2 = np.linspace(0, 1, 21)
            xx1,xx2 = np.meshgrid(x1,x2)
            xq = np.array([xx1.flatten(), xx2.flatten()])
            zz = kdei(xq).reshape(xx1.shape)
            ax.pcolormesh(xx1, xx2, zz, cmap=cmocean.cm.amp, vmin=0, rasterized=True)
            if t_true is not None:
                ax.plot(t_true[col], t_true[row], 'b+', markeredgewidth=2.5, markeredgecolor='w', markersize=10)
                ax.plot(t_true[col], t_true[row], 'b+', markeredgewidth=1.5, markeredgecolor='b', markersize=10)

            # zqq = np.quantile(zz.flatten(), 0.05)
            zordered = np.sort(zz.flatten())
            zsum = np.cumsum(zordered)
            zindex = np.argmin( (zsum - 0.05*zsum[-1])**2)
            zqq = zordered[zindex]
            # print('zindex:', zindex)
            # print('zsum:', zsum)
            # ax.contour(xx1, xx2, zz, levels=[zqq,], colors='k', linestyles='dashed', linewidths=0.5)
            
            if col==0:
                ax.text(-0.4, 0.5, t_names[row], rotation=90, va='center', ha='right')
            if row==7:
                ax.text(0.5, -0.4, t_names[col], rotation=0, va='top', ha='center')
            # ax.grid(linestyle=':')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.])
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.])
            
            r_test = scipy.stats.pearsonr(thetas[nburn:, col], thetas[nburn:, row])
            r = r_test.statistic
            # pvalue = r_test.pvalue
            # if pvalue
            axT = axs[col, row]
            axT.set_visible(True)
            textcolor = 'k' if np.abs(r)<0.8 else 'w'
            axT.text(0.5, 0.5, '{:.2f}'.format(r), transform=axT.transAxes, 
                ha='center', va='center', color=textcolor, fontweight='bold')
            axT.set_xticks([])
            axT.set_yticks([])
            axT.set_facecolor(cmocean.cm.balance(0.5*(1 +r)))
            
            # ax.set_xticks([0, 1], theta_bounds[col])
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1], 
                [theta_bounds[row][0], '', '', '', theta_bounds[row][1]])
            if col>0:
                ax.set_yticklabels([])
            if row<7:
                ax.set_xticklabels([])

        ax = axs[row,row]
        ax.set_visible(True)
        density = scipy.stats.gaussian_kde(thetas[nburn:, row])
        xpdf = np.linspace(0, 1, 21)
        pdf = density(xpdf)
        ax.axhline(1., color='k', linestyle='dashed', label='Prior')
        ax.plot(xpdf, pdf, label='Posterior', color='red')
        if t_true is not None:
            ax.axvline(t_true[row], label='Target', color='blue')
        ax.grid(linestyle=':')
        ax.yaxis.tick_right()
        ax.set_xlim([0, 1])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels([])
        ax.set_ylim([0, 5])
        ax.set_yticks([0, 2, 4])

    axs[0,0].legend(bbox_to_anchor=(0., 1.0, 0.5, 1), loc='lower left', frameon=False, ncols=3)
    axs[0,0].text(-0.4, 0.5, t_names[0], rotation=90, va='center', ha='right',
        transform=axs[0,0].transAxes)
    # axs[-1,-1].set_xlabel(t_names[-1])
    axs[-1,-1].text(0.5, -0.4, t_names[-1], rotation=0, va='top', ha='center',
        transform=axs[-1,-1].transAxes)
    for i,ax in enumerate(axs[-1, :]):
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], labels=[theta_bounds[i][0], '', '','', theta_bounds[i][1]], rotation=45)
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.075, top=0.95, wspace=0.3, hspace=0.2)
    # fig.savefig('figures/04_theta_posterior_distributions_m{:03d}_p{:02d}.png'.format(m, p), 
    #     dpi=400)
    figaxs.append((fig,axs))

    # PREDICTIONS
    smpl = model.get_samples(nburn=nburn, includelogpost=False)
    print('Drew {} posterior samples'.format(len(smpl['betaU'])))
    if not os.path.exists('data/pred'):
        os.makedirs('data/pred')
    if recompute:
        preds = SepiaEmulatorPrediction(samples=smpl, model=model, addResidVar=True)
        preds.w = preds.w.astype(np.float32)
        eta_pred = preds.get_y().astype(np.float32)
        np.save('data/pred/eta_m{:03d}_p{:02d}.npy'.format(m, p), eta_pred)
    else:
        eta_pred = np.load('data/pred/eta_m{:03d}_p{:02d}.npy'.format(m, p))
    y_mean = np.mean(eta_pred, axis=0).squeeze()
    print('Computing quantiles...')
    y_lower0 = np.quantile(eta_pred, 0.025, axis=0).squeeze()
    y_lower1 = np.quantile(eta_pred, 0.25, axis=0).squeeze()
    y_upper0 = np.quantile(eta_pred, 0.975, axis=0).squeeze()
    y_upper1 = np.quantile(eta_pred, 0.75, axis=0).squeeze()
    print('Done quantiles')
        
    prior_lower = np.quantile(Y_sim, 0.025, axis=0).squeeze()
    prior_upper = np.quantile(Y_sim, 0.975, axis=0).squeeze()

    print('Plotting calibrated model')
    fig, ax = plt.subplots(figsize=(6, 3))
    h1 = ax.plot(Y_ind_obs, Y_sim[:, :].T, color='gray', 
        linewidth=0.3, label='Ensemble', zorder=2, alpha=0.25)
    f0 = ax.fill_between(Y_ind_obs, prior_lower[:], prior_upper[:], 
        label='Prior 95% interval', zorder=1, color='gray', 
        edgecolor='none', alpha=0.25)
    h2 = ax.plot(Y_ind_obs, Y_obs.squeeze(), color='blue', 
        label=label, zorder=2)
    h3 = ax.plot(Y_ind_obs, y_mean[:].squeeze(), color='red', 
        label='Calibrated model mean', zorder=2)
    f1 = ax.fill_between(Y_ind_obs, y_lower0[:], y_upper0[:], 
        color='red', label='Calibrated model 95% interval', 
            alpha=0.3, zorder=1, edgecolor='none')
    # f2 = ax.fill_between(Y_ind_obs, y_lower1[:], y_upper1[:], 
    #     color='firebrick', label='Calibrated model 50% interval', 
    #     alpha=0.4, zorder=1, edgecolor='none')
    ax.grid(linestyle=':', zorder=0)
    ax.legend(handles=(h1[0], f0, h3[0], f1, h2[0]), 
        bbox_to_anchor=(-0.05,1,1,0.2), loc='lower left', frameon=False, ncols=3)
    ax.set_ylabel('Flotation fraction')
    ax.set_xlabel('Day of year')
    ax.set_xlim([Y_ind_obs[0], Y_ind_obs[-1]])
    ax.set_ylim([0, 2.])
    fig.subplots_adjust(bottom=0.15, top=0.8, left=0.11, right=0.975)
    # fig.savefig('figures/05_ypred_posterior_predictions_m{:03d}_p{:02d}.png'.format(m, p),
    #     dpi=600)
    print('Done calibrated model')
    figaxs.append((fig,ax))

    return figaxs
