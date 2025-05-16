import os
import sys

import pickle
import numpy as np
import os

import matplotlib
matplotlib.rc
fs = 8
matplotlib.rc('font', size=fs)
matplotlib.rc('axes', labelsize=fs)
matplotlib.rc('axes', titlesize=fs)
#matplotlib.use('QtAgg')
from matplotlib import pyplot as plt
from matplotlib import tri
from matplotlib.patches import Rectangle, Polygon
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib import cm
from matplotlib import colors as mpc
import cmocean
import rasterio as rs

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath

from utils import tools

# from palettes.code import palettes

# Geometry and compute misc fields
bed = np.vstack(np.load('../issm/issm/data/geom/IS_bed.npy'))
surf = np.vstack(np.load('../issm/issm/data/geom/IS_surface.npy'))
thick = surf - bed
bed[thick<50] = surf[thick<50] - 50
thick = surf - bed

aws_xy = [-217706.690013, -2504221.345267]
bh_xy = [-205722.455632, -2492714.204274]

# Compute triangulation
# with open('../issm/issm/data/geom/IS_mesh.pkl', 'rb') as meshin:
#     mesh = pickle.load(meshin)
mesh = np.load('../issm/issm/data/geom/IS_mesh.pkl', allow_pickle=True)
triangulation = tri.Triangulation(mesh['x']/1e3, mesh['y']/1e3, mesh['elements']-1)

# mesh['connect_edge'] = tools.reorder_edges_mesh(mesh)

# Moulins
with open('../issm/issm/data/yang2016_moulins/moulins_catchments_YS16.pkl', 'rb') as infile:
    basins = pickle.load(infile)
    moulin_indices = np.array([basin['moulin'] for basin in basins])

# Simulations
train_config = tools.import_config('../train_config.py')
train_config.m = 256
test_config = tools.import_config('../test_config.py')
# Read in the ensemble of simulations
Y_sim_phys = np.load(train_config.Y_physical, mmap_mode='r').T[:train_config.m]
Y_test = np.load(test_config.Y_physical, mmap_mode='r').T
S_sim_phys = np.load(train_config.Y_physical.replace('_ff', '_S'), mmap_mode='r').T[:train_config.m]
Q_sim_phys = np.load(train_config.Y_physical.replace('_ff', '_Q'), mmap_mode='r').T[:train_config.m]

n_edges = len(mesh['edge_length'])
n_t = 365
S_arr = S_sim_phys.T.reshape((n_edges, n_t, train_config.m))
# S_total = np.sum(S_arr*mesh['edge_length'][:, None, None], axis=(0,1))
# S_total = np.sum(S_arr, axis=(0,1))
ff_arr = Y_sim_phys.T.reshape((mesh['numberofvertices'], n_t, train_config.m))
Q_arr = Q_sim_phys.T.reshape((n_edges, n_t, train_config.m))
# print('S_total.shape:', S_total.shape)
# print(np.median(S_total))
# ff_total = np.mean(ff_arr, axis=(0,1))
# S_score = S_total
# ff_score = ff_total/np.max(ff_total)
# score = S_score
# m_median = np.argmin(score - np.median(score))
# print('m_median:', m_median)
m_median = 90

print('Starting tripcolor panels')
fig = plt.figure(figsize=(7, 6.5))

dx = 10
gs = GridSpec(2, 5,
    hspace=0.2, wspace=0.45, left=0.025, bottom=0.06, right=0.95, top=0.75,
    height_ratios=(5/7, 2/7), width_ratios=(dx, 100, dx, 100, dx))
ax1 = fig.add_subplot(gs[0,:-1])
ax2 = ax1.inset_axes((0.35, 0.75, 0.8, 0.8))
ax3 = ax1.inset_axes((0.0, 0.85, 0.35, 0.65))

axt1 = fig.add_subplot(gs[-1,1:3])
axt2 = fig.add_subplot(gs[-1,3:])


h3 = 0.9
cax = ax1.inset_axes((1.02, 0.1, 0.025, 0.63))
cax2 = ax1.inset_axes((0.4, 0.065, 0.35, 0.03))
pos = np.array([296, 34, 38])
tstep = 230
for ax in (ax1, ax2):
    ec = '#b0b0b0' if ax is ax2 else 'none'
    elw = 0.2 if ax is ax2 else 0.1
    sc = ax.tripcolor(triangulation, ff_arr[:, tstep, m_median],
        vmin=0, vmax=1, cmap=cmocean.cm.dense, edgecolor=ec,
        linewidth=elw, alpha=0.9, rasterized=True, antialiased=True)
    ax.tricontour(triangulation, surf.squeeze(), levels=(1700,), 
        # colors=[cmocean.cm.algae(0.4)], linestyle='dashed',
        colors=['#aaaaaa'], linestyles='dashed',
        # colors = cmocean.cm.algae(0.4), linestyles='dashed',
        )
    # ela.set_label('ELA (1700 m asl.)')
    
    if ax is ax1:
        Smin = 1
        Smax = 200
        cnorm = matplotlib.colors.Normalize(vmin=Smin, vmax=Smax)
        lscale = 1 if ax is ax1 else 0.3
        qlist = np.where(np.abs(Q_arr[:, :, m_median])>Smin)[0]
        lc_colors = []
        lc_lw = []
        lc_xy = []
        for i in qlist:
            Qi = np.abs(Q_arr[i, tstep, m_median])
            # if Qi>Smin:
            # ax.plot(mesh['x'][mesh['connect_edge'][i,:]]/1e3,
            #     mesh['y'][mesh['connect_edge'][i,:]]/1e3,
            #     linewidth=lscale*(0.25+1.25*cnorm(Qi)), 
            #     color=cmocean.cm.turbid(cnorm(Qi)))
            x0,x1 = mesh['x'][mesh['connect_edge'][i,:]]/1e3
            y0,y1 =  mesh['y'][mesh['connect_edge'][i,:]]/1e3
            lc_xy.append([(x0, y0), (x1, y1)])
            lc_lw.append(lscale*(0.25+1.25*cnorm(Qi)))
            lc_colors.append(cmocean.cm.turbid(cnorm(Qi)))
        lc = LineCollection(lc_xy, colors=lc_colors, linewidths=lc_lw,
            capstyle='round')
        ax.add_collection(lc)

    # sc = ax.tripcolor(triangulation, thick.flatten(), vmin=0, vmax=2500, cmap=cmocean.cm.ice_r,
    #     edgecolor='#aaaaaa', linewidth=0.1)
    if ax is ax2:
        cbar = fig.colorbar(sc, shrink=0.75, pad=0.02, cax=cax, orientation='vertical')
        cbar.set_label('Fraction of overburden')
        # cax.xaxis.tick_top()
        # cax.xaxis.set_label_position('top')
    if ax is ax1:
        ms = 10
        mlw = 2
    else:
        ms = 5
        mlw = 1
    ax.set_aspect('equal')
    ax.set_facecolor('none')
    if ax is ax1:
        handle_moulins = ax.plot(mesh['x'][moulin_indices]/1e3, mesh['y'][moulin_indices]/1e3,
            marker='o', color='w', markersize=ms/2, linestyle='', label='Moulins',
            markeredgecolor='k', linewidth=0.5)
    handle_outlets = ax.plot(mesh['x'][pos]/1e3, mesh['y'][pos]/1e3, linestyle='',
        marker='*', color='r', markersize=ms/1.5, label=r'$p_{\rm{w}}=0$ outlets')
    # ax.plot(aws_xy[0]/1e3, aws_xy[1]/1e3, '^', markersize=ms/1.5, label='KAN_L AWS', color='m')
    handle_borehole = ax.plot(bh_xy[0]/1e3, bh_xy[1]/1e3, '^', 
        markersize=ms, label='Borehole GL12-2A', color='b', 
        markeredgecolor='w')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['left', 'bottom', 'right', 'top']].set_visible(False)
    ax.set_facecolor('none')


rivers = ['Isotorq R.', 'Sandflugtsdalen R.', 'Sandflugtsdalen R.']
glaciers = ['IS', 'RG', 'LG']
ax1.text(mesh['x'][pos[0]]/1e3-2, mesh['y'][pos[0]]/1e3,
    'IS', va='bottom', ha='right')
ax1.text(mesh['x'][pos[1]]/1e3-2, mesh['y'][pos[1]]/1e3,
    'RG', va='bottom', ha='right')
ax1.text(mesh['x'][pos[2]]/1e3-2, mesh['y'][pos[2]]/1e3,
    'LG', va='top', ha='right')

cb2 = fig.colorbar(cm.ScalarMappable(norm=cnorm, cmap=cmocean.cm.turbid),
    cax=cax2, orientation='horizontal')
cb2.set_ticks([Smin, 50, 100, 150, 200])
cb2.set_label('Channel discharge (m$^3$ s$^{-1}$)')
# ax1.set_aspect('equal')
zmax = 1850
surf0 = surf[:, 0]
xmax = np.max(mesh['x'][surf0<=zmax])/1e3
ymax = np.max(mesh['y'][surf0<=zmax])/1e3
xmin = np.min(mesh['x'][surf0<=zmax])/1e3
ymin = np.min(mesh['y'][surf0<=zmax])/1e3
ax1.set_xlim([xmin-1, xmax])
ax1.set_ylim([ymin, ymax])

ax2.set_xlim([xmin-2, np.max(mesh['x'])/1e3])

print('Length of area below zmax:', xmax - xmin)

rect = Rectangle(xy=(xmin, ymin), width=(xmax-xmin), height=(ymax-ymin))
pc = PatchCollection([rect], facecolor='none', edgecolor='k',
    linestyle=':', linewidth=1)
ax2.add_collection(pc)


scale = Rectangle(xy=(xmin+0, ymin+0.065*(ymax-ymin)), width=50, height=1, zorder=15)
spc = PatchCollection([scale], facecolor='k', clip_on=False)
ax1.add_collection(spc)
ax1.text(xmin+0+0.5*50, ymin, '50 km', ha='center', va='bottom')

_phony = LineCollection([((0, 1), (0, 1))], color='#aaaaaa', label='ELA (1700 m asl.)', linestyle='dashed')
print(handle_moulins)
print(handle_outlets)
print(handle_borehole)
print(_phony)
ax1.legend(
    handles=[handle_moulins[0], handle_outlets[0], handle_borehole[0], _phony],
    bbox_to_anchor=(0, 0.15, 0.5, 0.8), 
    frameon=False, loc='lower left', borderpad=0, borderaxespad=0,
    markerscale=2)



# scale2 = Rectangle(xy=(mesh['x'].max()/1e3-10-100, ymax-20), width=100, height=2, zorder=15)
# spc2 = PatchCollection([scale2], facecolor='k', clip_on=False)
# ax2.add_collection(spc2)
# ax2.text(mesh['x'].max()/1e3-10-0.5*100, ymax-20+5+2, '100 km', ha='center', va='bottom')

print('Starting inset contour')
with rs.open('../issm/issm/data/geom/GimpIceMask_90m_2015_v1.2.tif') as geotiff:
    raster_mask = geotiff.read(1)

    rxmin = geotiff.bounds.left
    rxmax = geotiff.bounds.right
    rymin = geotiff.bounds.bottom
    rymax = geotiff.bounds.top
    nrows,ncols = geotiff.shape
    
x = np.linspace(rxmin, rxmax, ncols+1)[:-1]/1e3
y = np.linspace(rymin, rymax, nrows+1)[0:-1][::-1]/1e3
[xx, yy] = np.meshgrid(x, y)

inc = 10
ax3.contour(xx[::inc,::inc], yy[::inc,::inc], raster_mask[::inc,::inc], 
    levels=(0,), colors='k', linewidths=0.35)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.spines[['left', 'bottom', 'right', 'top']].set_visible(False)
ax3.set_aspect('equal')
outline = np.loadtxt('../issm/issm/data/geom/IS_outline.csv', skiprows=1, quotechar='"', delimiter=',')
# ax3.plot(outline[:, 1]/1e3, outline[:, 2]/1e3)
xy = np.array([outline[:, 1], outline[:, 2]]).T
ax3.tripcolor(triangulation, thick.flatten(), vmin=0, vmax=2500, cmap=cmocean.cm.ice_r,
        edgecolor='none', rasterized=True)
pg = Polygon(xy[::10]/1e3, closed=True, facecolor='none', edgecolor='b', linewidth=1)
ax3.add_patch(pg)
ax3.set_facecolor('none')
# ax3.plot(aws_xy[0]/1e3, aws_xy[1]/1e3, '^', markersize=3, color='m')

## PLOT DATA
# Read in the simulation input settings
t_sim = np.loadtxt(train_config.X_standard, delimiter=',', skiprows=1)[:train_config.m]
t_names = train_config.theta_names
dx = mesh['x'] - bh_xy[0]
dy = mesh['y'] - bh_xy[1]
# nodenum = np.argmin(dx**2 + dy**2)
nodenum = 3611
print('Extracting ensemble at node', nodenum)

surf = np.load(os.path.join(train_config.sim_dir, '../data/geom/IS_surface.npy'))
bed = np.load(os.path.join(train_config.sim_dir, '../data/geom/IS_bed.npy'))
thick = surf - bed

# BOREHOLE DATA
print('Loading borehole data')
Y_obs_phys_dataset = np.loadtxt('processed/GL12-2A_Pw_daily_2012.txt',
    skiprows=0, delimiter=',')

bh_tt = Y_obs_phys_dataset[:, 0] - 2
bh_ff = Y_obs_phys_dataset[:, 1]
t0 = bh_tt[0]
t1 = 365
bh_ff = bh_ff[np.logical_and(bh_tt>=t0, bh_tt<t1)]
bh_tt = bh_tt[np.logical_and(bh_tt>=t0, bh_tt<t1)]

# tt = 2012 + np.arange(365)/365
tt = np.arange(365)

ti0 = np.where(np.abs(tt-bh_tt[0])<1e-10)[0]
ti1 = np.where(np.abs(tt-bh_tt[-1])<1e-10)[0]
print(ti0, ti1)
y_ind = nodenum*365 + np.arange(365)

# Find an appropriate candidate
# t_target = np.array([1./3., 2./3., 0.5, 0.5, 0.5, 2./3., 0.5, 0.5])
# distance = np.sum((t_sim-t_target)**2, axis=1)
# sim_num = np.argmin(distance)
# print(sim_num, t_sim[sim_num])
sim_num = 90

Y_sim_phys = Y_sim_phys[:, y_ind]
Y_test = Y_test[:, y_ind]
Y_obs_sim = Y_test[sim_num,:]
print('Y_sim_phys:', Y_sim_phys.shape)
print(Y_obs_sim.shape)

h2 = axt1.plot(tt, Y_obs_sim, color='b', zorder=3,
    label='Data')
axt2.plot(bh_tt, bh_ff, color='b', zorder=3)

for ax in (axt1, axt2):
    h1 = ax.plot(tt, Y_sim_phys.T, color='gray', linewidth=0.3, alpha=0.3, zorder=2,
        label='Ensemble')
    ax.grid(linewidth=0.5)
    ax.set_ylim([0, 3])
    ax.set_yticks(np.linspace(0, 3, 7))
    # ax.set_xlim([2012, 2013])
    # ax.set_xlim([120, 300])
    ax.set_xlim([0, 365])
    ax.set_xlabel('Day of 2012')
    ax.axvline(tt[tstep], color='k', linestyle='dashed')

axt2.set_ylim([0.8, 1.05])
axt2.set_yticks([0.8, 0.85, 0.9, 0.95, 1.0, 1.05])

axt1.legend(handles=(h1[0], h2[0]), loc='upper right', frameon=True)
# axt2.set_yticklabels([])
axt1.set_ylabel('Fraction of overburden')

axt1.set_title('Synthetic data')
axt1.text(0.025, 0.95, '(d)', transform=axt1.transAxes,
    fontweight='bold', va='top')
axt2.set_title('Borehole data')
axt2.text(0.025, 0.95, '(e)', transform=axt2.transAxes,
    fontweight='bold', va='top')



fig.text(0.025, 0.975, '(a)',
    fontweight='bold', va='top', ha='left')

fig.text(0.325, 0.975, '(b)',
    fontweight='bold', va='top', ha='left')

ax2.text(xmin+2, ymin+2, '(c)', va='bottom', ha='left', zorder=10)
# ax1.plot(xmaymin+1, 'rx')
# print(xmin, xmax)

fig.text(0.025, 0.675, '(c)', fontweight='bold',
    va='bottom', ha='left')

fig.savefig('greenland_domain_summary.png', dpi=400)
fig.savefig('greenland_domain_summary.pdf', dpi=400)
