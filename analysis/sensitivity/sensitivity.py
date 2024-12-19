import numpy as np
import matplotlib
matplotlib.rc('font', size=8)
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.legend_handler import HandlerTuple
import scipy.stats
import cmocean

Y_ref = np.load('../../issm/issm/train_numerics/greenland_ff.npy', mmap_mode='r')
Y_s1 = np.load('../../issm/issm/S01_creepopen/greenland_ff.npy', mmap_mode='r')
Y_s2 = np.load('../../issm/issm/S02_nomoulins/greenland_ff.npy', mmap_mode='r')
Y_s3 = np.load('../../issm/issm/S03_turbulent/greenland_ff.npy', mmap_mode='r')
cases = ['Ref', 'Creep open', 'No moulins', 'Turbulent']

bh_config = np.load('../../GL12-2A.pkl', allow_pickle=True)
nodenum = bh_config['node']
bh_record = np.loadtxt(bh_config['path'], delimiter=',')
# obs_days, Y_obs = bh_record[:199].T  # Correct for leap day missing from the model and make zero-indexed
obs_days, Y_obs = bh_record[:199].T
Y_obs = Y_obs.astype(np.float32)
# Y_obs = 0.9 * Y_obs
# print('Raw obs_days:', obs_days)
# Correct for leap day missing from the model and make zero-indexed
obs_days = (obs_days - 2).astype(int)
# print('Processed obs_days:', obs_days)
y_ind_obs = (nodenum*365 + obs_days).astype(int)
y_ind_sim = nodenum*365 + np.arange(365)

tt = np.arange(365)
yind = nodenum*365 + tt
wind = 140
Y_ref = Y_ref[yind, :]
Y_s1 = Y_s1[yind, :]
Y_s2 = Y_s2[yind, :]
Y_s3 = Y_s3[yind, :]

fig, axs = plt.subplots(figsize=(6, 4), nrows=2)
cmap = cmocean.cm.delta
# cmap = matplotlib.colormaps['RdYlBu']
colors = ('k', cmap(0.2), cmap(0.6), cmap(0.8))
hatches = ('', '', '', '')
zorder = (2, 3, 2, 1)
alpha = (0.2, 0.4, 0.4, 0.2)
Yvals = (Y_ref, Y_s1, Y_s2, Y_s3)
handles = []
for ax in axs:
    for i in range(len(Yvals)):
        p1 = ax.plot(tt, np.median(Yvals[i][:, :64], axis=1),
            color=colors[i], zorder=3, label=cases[i], linewidth=2)
        p2 = ax.fill_between(tt, np.quantile(Yvals[i][:, :64], 0.25, axis=1),
            np.quantile(Yvals[i][:, :64], 0.75, axis=1), color=colors[i], alpha=alpha[i],
            edgecolor='none', zorder=zorder[i], hatch=hatches[i], linewidth=10,label=cases[i],
        )
        handles.append((p1[0], p2))
    ax.legend(handles, cases, loc='upper right', 
        handler_map={tuple: HandlerTuple(ndivide=None)}, handlelength=5)
    ax.grid(linewidth=0.5)
    ax.set_ylabel('Flotation fraction')
    ax.set_xlabel('Day of 2012')
    
axs[0].set_xlim([140, 300])
axs[0].set_ylim([0.5, 2.25])
axs[1].set_xlim([140, 190])
axs[1].set_ylim([0.5, 2.25])
fig.subplots_adjust(bottom=0.1, top=0.95, right=0.95, left=0.1, hspace=0.25)
fig.savefig('figures/sensitivity_pw.png', dpi=400)

cmap = cmocean.cm.delta
cases_rmse = []
fig, axs = plt.subplots(figsize=(6, 7), nrows=4, sharex=True)
for i in range(len(Yvals)):
    ax = axs[i]
    handles = []
    # p1 = ax.plot(tt, np.median(Yvals[i][:, :64], axis=1),
    #     color=colors[i], zorder=3, label=cases[i], linewidth=2)
    err = Yvals[i][obs_days, :64] - Y_obs[:,None]
    rmse = np.sqrt(np.mean(err**2, axis=0))
    cases_rmse.append(np.min(rmse))
    amin = np.argmin(rmse)
    p1 = ax.plot(tt,Yvals[i][:, amin],
        color=colors[i], zorder=3, label=cases[i], linewidth=2)
    
    p2 = ax.fill_between(tt, np.quantile(Yvals[i][:, :64], 0.05, axis=1),
        np.quantile(Yvals[i][:, :64], 0.975, axis=1), color=colors[i], alpha=alpha[i]/2,
        edgecolor='none', zorder=zorder[i], hatch=hatches[i], linewidth=10,label=cases[i],
    )
    handles.append((p1[0], p2))
    h = ax.plot(obs_days, Y_obs, 'b', label='Borehole GL12-2A')
    handles.append(h[0])
    ax.set_xlim([160, 300])
    ax.set_ylim([0.8, 1.1])
    ax.legend(handles, [cases[i], 'Borehole GL12-2A'],
        bbox_to_anchor=(0,1,1,0.5), loc='lower left', ncols=3, frameon=False,
        handler_map={tuple: HandlerTuple(ndivide=None)}, handlelength=5)
    ax.grid(linewidth=0.5)
    ax.set_ylabel('Flotation fraction')
    ax.text(0.975, 0.95, 'rmse = {:.3f}'.format(np.min(rmse)),
            ha='right', va='top', transform=ax.transAxes)
ax.set_xlabel('Day of 2012')
fig.subplots_adjust(left=0.1, bottom=0.0625, top=0.95, right=0.95)
fig.savefig('figures/sensitivity_cases.png'.format(i), dpi=400)

print(cases)
print(cases_rmse)

fig,axs = plt.subplots(figsize=(4, 3))
winter = (Y_ref[wind,:], Y_s1[wind,:], Y_s2[wind,:], Y_s3[wind,:])
xx = np.linspace(0, 1, 51)
bins = np.linspace(0, 1, 11)
# colors = ['gray', 'tab:blue', 'tab:orange', 'tab:green']
for i,y in enumerate(winter):
    # axs.hist(y, color=colors[i], label=cases[i], histtype='step', 
    #     bins=bins, linewidth=2, density=True)
    kd = scipy.stats.gaussian_kde(y)
    axs.plot(xx, kd(xx), color=colors[i], label=cases[i], linewidth=2)
    axs.fill_between(xx, 0*xx, kd(xx), color=colors[i], alpha=0.1)
axs.legend(loc='upper left')
axs.grid(linewidth=0.5)
axs.set_xlim([0, 1])
axs.set_ylim([0, 4])
axs.set_xlabel('Fraction of overburden')
axs.set_ylabel('Density')
axs.set_title('Winter water pressure')
fig.subplots_adjust(bottom=0.15, left=0.125, right=0.95, top=0.9)
fig.savefig('figures/sensitivity_winter_WP.png', dpi=400)

fig,axs = plt.subplots(figsize=(4,3))
summer = (np.max(Y_ref[:,:64], axis=0),
          np.max(Y_s1[:,:], axis=0),
          np.max(Y_s2[:,:], axis=0),
          np.max(Y_s3[:,:], axis=0),
         )
xx = np.linspace(0, 4, 51)
bins = np.linspace(0, 4, 21)
for i,y in enumerate(summer):
    # kd = scipy.stats.gaussian_kde(y)
    # axs.plot(xx, kd(xx), color=colors[i], label=cases[i])
    # axs.hist(y, color=colors[i], label=cases[i], histtype='step', 
    #     bins=bins, linewidth=2, density=True)
    kd = scipy.stats.gaussian_kde(y)
    axs.plot(xx, kd(xx), color=colors[i], label=cases[i], linewidth=2)
    axs.fill_between(xx, 0*xx, kd(xx), color=colors[i], alpha=0.1)
axs.legend(loc='upper right')
axs.grid(linewidth=0.5)
# axs.set_xlim([0, 1])
# axs.set_ylim([0, 4])
axs.set_xlabel('Fraction of overburden')
axs.set_ylabel('Density')
axs.set_title('Max summer water pressure')
fig.subplots_adjust(bottom=0.15, left=0.125, right=0.95, top=0.9)
fig.savefig('figures/sensitivity_summer_WP.png', dpi=400)