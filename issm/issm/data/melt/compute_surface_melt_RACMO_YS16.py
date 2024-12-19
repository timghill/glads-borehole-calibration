"""
Integrate surface melt within catchments
"""

import os
import sys
import pickle

import numpy as np


import cmocean
from matplotlib import pyplot as plt
import matplotlib

mesh = np.load('../geom/IS_mesh.pkl', allow_pickle=True)

runoff_time = np.load('RACMO_mesh_runoff.npy')
runoff = runoff_time[:-1]
tt = runoff_time[-1]

print('runoff.shape:', runoff.shape)

rho_water = 1000
node_melt = runoff/rho_water
element_melt = np.mean(node_melt[mesh['elements']-1], axis=1)

print('element_melt:', np.sum(element_melt))

# Read catchments
with open('../yang2016_moulins/moulins_catchments_YS16.pkl', 'rb') as fid:
    basins = pickle.load(fid)

area_ele = mesh['area']

n_basins = len(basins)
n_time = runoff.shape[1]
surf_inputs = np.zeros((n_basins, n_time))
for i in range(n_basins):
    areas = area_ele[basins[i]['elements']]
    meltrate = element_melt[basins[i]['elements']]
    basin_inputs = np.sum(np.vstack(areas)*meltrate, axis=0)
    surf_inputs[i,:] = basin_inputs

surf_inputs[surf_inputs<0] = 0
[xx, ll] = np.meshgrid(tt, np.arange(len(area_ele)))
[_, bb] = np.meshgrid(tt, np.arange(n_basins))

fig, ax = plt.subplots()
pc = ax.pcolormesh(xx, ll, element_melt*86400, cmap=cmocean.cm.rain, vmin=0, vmax=0.1)
cbar = fig.colorbar(pc)
cbar.set_label('Melt rate (m w.e./day)')
ax.set_title('Element melt rates')
ax.set_xlabel('Day of year')
ax.set_ylabel('Element index')
fig.savefig('element_melt_rates_RACMO_YS16.png', dpi=400)

fig, ax = plt.subplots()
pc = ax.pcolormesh(tt, bb, surf_inputs, cmap=cmocean.cm.rain, vmin=0, vmax=80)
cbar = fig.colorbar(pc)
cbar.set_label('Surface inputs (m$^3$/s)')
ax.set_title('Integrated catchment inputs')
ax.set_xlabel('Day of year')
ax.set_ylabel('Basin index')
fig.savefig('basin_integrated_inputs_RACMO_YS16.png', dpi=400)

issm_inputs = np.zeros((surf_inputs.shape[0]+1, surf_inputs.shape[1]))
issm_inputs[:-1] = surf_inputs
issm_inputs[-1, :] = tt
print('issm_inputs', issm_inputs[-1])
print(np.sum(surf_inputs))
np.save('basin_integrated_inputs_RACMO_YS16.npy', issm_inputs)
