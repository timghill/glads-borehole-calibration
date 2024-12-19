
import os
import sys
import pickle

import numpy as np


import cmocean
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation

# CONSTANTS
rho_i = 910         # Density of ice (kg/m3)
rho_w = 1000        # Density of water (kg/m3)
g = 9.81            # Gravity (m/s2)
L = 334e3           # Latent heat of fusion (J/kg)
target_melt = 0.04  # Melt rate (m w.e./a)

# READ FIELDS
surf = np.load('../geom/IS_surface.npy')
bed = np.load('../geom/IS_bed.npy')
thickness = surf - bed

slope = np.load('../geom/IS_surface_slope.npy')

# Upper bound to basal velocity
max_vel = np.load('../velocity/IS_surface_velocity.npy')

# COMPUTE BASAL MELT
drivingstress = rho_i*g*slope*thickness
# Upper bound to basal melt
max_basalmelt = max_vel * drivingstress / rho_w/L

# Scale basal velocity to achieve the correct max melt rate
vel_scalefactor = target_melt/np.max(max_basalmelt)
print('Scaling basal velocity by {:.6f}'.format(vel_scalefactor))
vel = max_vel * vel_scalefactor
np.save('IS_basal_velocity.npy', vel)

basalmelt = vel*drivingstress/rho_w/L
np.save('IS_friction_basal_melt.npy', basalmelt)

# PLOTS
with open('../geom/IS_mesh.pkl', 'rb') as meshin:
    mesh = pickle.load(meshin)

mtri = Triangulation(mesh['x']/1e3, mesh['y']/1e3, mesh['elements']-1)
fig, axs = plt.subplots(figsize=(6.5, 8), nrows=3, sharex=True)
ax1,ax2,ax3 = axs

tripc = ax1.tripcolor(mtri, vel, cmap=cmocean.cm.speed, vmin=0)
ax1.set_aspect('equal')
cbar = fig.colorbar(tripc)
cbar.set_label('Estimated basal velocity (m a$^{-1}$)')

tripc = ax2.tripcolor(mtri, drivingstress/1e3, cmap=cmocean.cm.rain, vmin=0)
ax2.set_aspect('equal')
cbar = fig.colorbar(tripc)
cbar.set_label('Driving stress (KPa)')


tripc = ax3.tripcolor(mtri, basalmelt, cmap=cmocean.cm.thermal, vmin=0, vmax=target_melt)
ax3.set_aspect('equal')
cbar = fig.colorbar(tripc)
cbar.set_label('Basal melt (m w.e. a$^{-1}$)')

fig.subplots_adjust(left=0.025, right=0.95, bottom=0.1, top=0.95, hspace=0.1)
fig.savefig('basal_melt.png', dpi=400)
