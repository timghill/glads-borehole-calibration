import numpy as np
import matplotlib
matplotlib.rc('font', size=8)
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import cmocean

table = np.loadtxt('table.dat')
# runs = np.array([1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

# runs = np.array([1, 2, 3, 6, 7, 8, 9, 10, 11, 12])
runs = np.arange(1, 21)
print('runs:', runs)
# failed = np.arange(1, 6)

num_nonconverge = [504,395,339,128,11,376,280,162,43,1,112,75,45,8,0,0,0,0,0,0]
runtime = [0.68,0.45,0.33,0.16,0.04,0.53,0.36,0.21,0.09,0.07,0.37,0.26,0.19,0.14,0.12,0.51,0.40,0.35,0.31,0.31]
refindex = 19

node = 3611
simnum = 91
tol = table[:, 1]
print('tol:', tol)
dt = table[:, 2]

Y = np.zeros((20, 365))
for i,run in enumerate(runs):
    yi = np.load('RUN/output_{:03d}/ff.npy'.format(run), mmap_mode='r')
    yi = yi[node, -366:-1]
    Y[run-1] = yi

# for i in failed:
#     Y[i-1] = np.nan

dt_linestyle = ['solid', 'dashed', 'dotted', 'dashdot']
tol_color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

itol,idt = np.meshgrid(np.arange(5), np.arange(4)[::-1])
itol = itol.flatten()
idt = idt.flatten()

print('itol:', itol)
print('idt:', idt)

fig,ax = plt.subplots(figsize=(8,4))
for i in range(20):
    ls = dt_linestyle[idt[i]]
    color = tol_color[itol[i]]
    ax.plot(np.arange(365), Y[i],
        label=i+1, color=color, ls=ls)

ax.legend()


dy = Y - Y[refindex, :]
rmse = np.sqrt(np.mean((dy**2), axis=1))
print(rmse.shape)
fig,ax = plt.subplots()
ax.scatter(dt, rmse)
ax.set_xlabel('dt')
ax.set_xscale('log')

fig,ax = plt.subplots()
ax.scatter(tol, rmse)
ax.set_xlabel('tol')
ax.set_xscale('log')

fig,axs = plt.subplots(figsize=(8,8), nrows=3, sharex=True)
for i in range(20):
    toli = table[i,1]
    dti = table[i,2]
    ls = 'solid'
    color = tol_color[itol[i]]
    if dti==0.2:
        axs[2].plot(np.arange(365), Y[i],
            label='restol = {:.2e} ({})'.format(toli, num_nonconverge[i]), color=color, ls=ls)
    elif dti==0.5:
        axs[1].plot(np.arange(365), Y[i],
            label='restol = {:.2e} ({})'.format(toli, num_nonconverge[i]), color=color, ls=ls)
    elif dti==1.0:
        axs[0].plot(np.arange(365), Y[i],
            label='restol = {:.2e} ({})'.format(toli, num_nonconverge[i]), color=color, ls=ls)
for ax in axs:
    ax.legend()
    ax.set_xlim([0, 365])
    ax.set_ylim([0.6, 1.9])
    ax.grid()

axs[0].set_title('dt = 1.0')
axs[1].set_title('dt = 0.5')
axs[2].set_title('dt = 0.2')
fig.subplots_adjust(bottom=0.1, left=0.05, right=0.95, top=0.95)
axs[-1].set_xlabel('Day of 2012')
fig.savefig('numerical_convergence.png', dpi=400)

mesh = np.load('../data/geom/IS_mesh.pkl', allow_pickle=True)
nx = mesh['numberofvertices']
Y = np.zeros((20, nx, 365))
for i,run in enumerate(runs):
    yi = np.load('RUN/output_{:03d}/ff.npy'.format(run))
    Y[run-1] = yi[:, -366:-1]
fig,ax = plt.subplots(figsize=(8,4))
std = np.mean(np.std(Y[5:], axis=(0)), axis=-1)
print(std.shape)

mtri = Triangulation(mesh['x'], mesh['y'], mesh['elements']-1)
zmax = 1700
surf = np.load('../data/geom/IS_surface.npy')
ela = 1850
xmin = np.min(mesh['x'][surf<=ela]) - 5e3
xmax = np.max(mesh['x'][surf<=ela])
ymin = np.min(mesh['y'][surf<=ela])
ymax = np.max(mesh['y'][surf<=ela])
pc = ax.tripcolor(mtri, std, vmin=0, vmax=0.05, cmap=cmocean.cm.amp)
ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))
ax.set_aspect('equal')
ax.plot(mesh['x'][node], mesh['y'][node], 'b^')
cbar = fig.colorbar(pc, extend='max')
cbar.set_label('Flotation fraction standard deviation')
fig.subplots_adjust(left=0.075, bottom=0.05, right=1., top=0.95)
fig.savefig('numerical_convergence_nodes.png', dpi=400)

plt.show()