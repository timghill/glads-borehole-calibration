import numpy as np
import pickle

import os
import sys
ISSM_DIR = os.getenv('ISSM_DIR')
sys.path.append(os.path.join(ISSM_DIR, 'bin/'))
sys.path.append(os.path.join(ISSM_DIR, 'lib/'))
from issmversion import issmversion
from hydrologyglads import hydrologyglads
from generic import generic
from timesteppingadaptive import timesteppingadaptive

# Vectors for convenience
xvec = md.mesh.x - np.min(md.mesh.x)
onevec = 0*xvec + 1

# Calving
md.calving.calvingrate=0*onevec

# # Friction - need to specify but not used
md.friction.coefficient = onevec
md.friction.p = np.ones((md.mesh.numberofelements, 1))
md.friction.q = np.ones((md.mesh.numberofelements, 1))

# Geometry
bed = np.load('../data/geom/IS_bed.npy')
surf = np.load('../data/geom/IS_surface.npy')
thick = surf - bed
thick[thick<10] = 10
surf = bed + thick
md.geometry.base = bed
md.geometry.bed = bed
md.geometry.surface = surf
md.geometry.thickness = thick

# Constants
md.materials.rheology_B = (2.4e-24)**(-1/3)*onevec
md.initialization.temperature = (273)*onevec
md.materials.rheology_n = 3
md.materials.rho_freshwater = 1e3
md.materials.rho_ice = 910
md.materials.mu_water = md.materials.rho_freshwater * 1.793e-6
md.constants.g = 9.8

# HYDROLOGY
# parameters
md.hydrology = hydrologyglads()
md.hydrology.sheet_conductivity = 0.05*onevec
md.hydrology.sheet_alpha = 3./2.
md.hydrology.sheet_beta = 2.0
md.hydrology.cavity_spacing = 10
md.hydrology.bump_height = 0.5*onevec
md.hydrology.channel_sheet_width = 50
md.hydrology.omega = 1/2000
md.hydrology.englacial_void_ratio = 1e-4
md.hydrology.rheology_B_base = (2.4e-24)**(-1./3.)*onevec
md.hydrology.istransition = 1
md.hydrology.ischannels = 1
md.hydrology.channel_conductivity = 0.5*onevec
md.hydrology.channel_alpha = 5./4.
md.hydrology.channel_beta = 3./2.
md.hydrology.creep_open_flag = 0
md.hydrology.isincludesheetthickness = 1
md.hydrology.requested_outputs = [
        'HydraulicPotential',
        'EffectivePressure',
        'HydrologySheetThickness',
        'ChannelDischarge',
        'ChannelArea',
        # 'HydrologySheetDischarge',
        'HydrologyWaterVx',
        'HydrologyWaterVy',
]

# INITIAL CONDITIONS
md.initialization.watercolumn = 0.2*md.hydrology.bump_height*onevec
md.initialization.channelarea = 0*np.zeros((md.mesh.numberofedges, 1))

phi_bed = md.constants.g*md.materials.rho_freshwater*md.geometry.base
p_ice = md.constants.g*md.materials.rho_ice*md.geometry.thickness
md.initialization.hydraulic_potential = phi_bed + p_ice

vv = np.load('../data/velocity/IS_basal_velocity.npy')
md.initialization.ve = vv
md.initialization.vx = -md.initialization.ve*onevec
md.initialization.vy = 0*onevec

# BOUNDARY CONDITIONS
md.hydrology.spcphi = np.nan*onevec
pos = np.array([296, 34, 38])
md.hydrology.spcphi[pos] = phi_bed[pos] + 0*p_ice[pos]
md.hydrology.neumannflux = np.zeros((md.mesh.numberofelements, 1))

# FORCING
md.hydrology.melt_flag = 1

# Sliding friction
# Set max friction melt to 4 cm/year (Harper et al., 2021)
# https://doi.org/10.5194/tc-15-5409-2021
# Scale friction melt with squared velocity, assuming basal drag
# is linear with respect to velocity (Sommers et al., 2023)
max_drag_melt = 0.04
basal_drag_melt = np.load('../data/velocity/IS_friction_basal_melt.npy')

# Geothermal
geo_flux_min = 27e-3
geo_flux_max = 49e-3
z_score = (md.geometry.surface - np.min(md.geometry.surface))/np.ptp(md.geometry.surface)
geo_flux = geo_flux_min + (geo_flux_max-geo_flux_min)*z_score
geo_melt = geo_flux/1e3/3.34e5 * md.constants.yts # Geothermal heat flux melt rate (m/a)
md.basalforcings.groundedice_melting_rate = basal_drag_melt + geo_melt
md.basalforcings.geothermalflux = 0

print('Min groundedice_melting_rate:', np.min(md.basalforcings.groundedice_melting_rate))
print('Max groundedice_melting_rate:', np.max(md.basalforcings.groundedice_melting_rate))

with open('../data/yang2016_moulins/moulins_catchments_YS16.pkl', 'rb') as moulins_file:
    basins = pickle.load(moulins_file)
moulin_indices = np.array([basin['moulin'] for basin in basins])
moulin_inputs = np.load('../data/melt/basin_integrated_inputs_RACMO_YS16.npy')
md.hydrology.moulin_input = np.zeros((md.mesh.numberofvertices+1, moulin_inputs.shape[1]))
md.hydrology.moulin_input[moulin_indices, :] = moulin_inputs[:-1, :]
md.hydrology.moulin_input[-1] = moulin_inputs[-1,:]

# TOLERANCES
md.stressbalance.restol = 1e-5
md.stressbalance.reltol = np.nan
md.stressbalance.abstol = np.nan
md.stressbalance.maxiter = 100

# # TIMESTEPPING
hour = 3600
day = 86400
dt_hours = 0.2
out_freq = 24*1/dt_hours
md.timestepping.time_step = dt_hours*hour/md.constants.yts
#md.timestepping.start_time = 2011.9
#md.timestepping.final_time = 2012.2
md.timestepping.start_time = 2010
md.timestepping.final_time = 2013
md.settings.output_frequency = out_freq

md.transient.deactivateall()
md.transient.ishydrology = True

md.verbose.solution = True
md.miscellaneous.name = 'output'

md.cluster = generic('np', 1)

SLURM_TMPDIR = os.getenv('SLURM_TMPDIR')
if SLURM_TMPDIR:
    md.cluster.executionpath = SLURM_TMPDIR
else:
    cwd = os.getcwd()
    expath = os.path.join(cwd, 'TMP/')
    if not os.path.exists(expath):
        os.makedirs(expath)
    md.cluster.executionpath = expath

print(md.cluster.executionpath)
