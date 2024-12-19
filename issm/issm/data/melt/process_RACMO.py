import xarray
import numpy as np
import scipy
from scipy import interpolate
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import cmocean

runoffFile = 'RACMO/Daily/runoff.RACMO2.3p2.FGRN055.2010-2014.DD.nc'
runoffDataset = xarray.open_dataset(runoffFile, engine='netcdf4')
meshFile = '../geom/IS_mesh.pkl'

maskFile = 'RACMO/FGRN055_Masks.nc'
maskDataset = xarray.open_dataset(maskFile, engine='netcdf4')
X = maskDataset.X
Y = maskDataset.Y

surfFile = '../geom/IS_surface.npy'
surface = np.load(surfFile)

mesh = np.load(meshFile, allow_pickle=True)

time = runoffDataset.time.data

# Fix an error
time[9] = np.datetime64('2010-01-10')

# Sort out time, removing leap days since ISSM doesn't do these properly
year_ref = 2000
tref = np.datetime64('{:d}'.format(year_ref), 'D')
year_int = year_ref + (time.astype('datetime64[Y]') - tref).astype('timedelta64[Y]').astype(int)
year = time.astype('datetime64[Y]')
day = time.astype('datetime64[D]')
dt = day - year
nextyear = np.array([np.datetime64('{}-01-01'.format(y+1)) for y in year_int])
numdays = (nextyear - year).astype(int)
print('numdays:', numdays)
print('day:', day.astype(int))
tyear = year_int + dt.astype(int)/numdays
print('dt:', dt)
print('num timesteps:', len(tyear))
#print('tyear:', tyear)
tyear = []
to_skip = np.zeros(len(time))
for i in range(len(time)):
    # Normal year case
    if numdays[i]==365:
        tyear.append(year_int[i] + dt[i].astype(int)/365)
    elif numdays[i]==366:
        if dt[i].astype(int)>0:
            tyear.append(year_int[i] + (dt[i].astype(int)-1)/365)
        else:
            # skip_indices.append(i)
            to_skip[i] = i

print('TO SKIP:', np.where(to_skip>0)[0])
tyear = np.array(tyear)
tindices = np.arange(len(time))[to_skip==0]
print('tyear:', tyear)
print('tindices:', tindices)
print(len(tindices))

print('Num timesteps:', len(tyear))
print('Num unique timesteps:', len(np.unique(tyear)))

runoff = runoffDataset.runoff
latCoords = runoffDataset.rlat
lonCoords = runoffDataset.rlon

# Find corners of the dataset
xd = X.data
yd = Y.data
print('xd:', xd.shape)
print('yd:', yd.shape)
maskx = np.logical_and(xd>=mesh['x'].min()/1e3, xd<=mesh['x'].max()/1e3)
masky = np.logical_and(yd>=mesh['y'].min()/1e3, yd<=mesh['y'].max()/1e3)
mask = np.logical_and(maskx, masky)
print('mask:', mask.shape)
Xm = xd[mask]
Ym = yd[mask]
rm = runoff[230, 0, :, :].data[mask]
rall = runoff[tindices, 0, :, :].data[:, mask]
tmax = int(np.where(tyear==2013)[0][0])
print('Last t index:', tmax)
rall = rall[:tmax]

fig,ax = plt.subplots(figsize=(8,4))

mtri = Triangulation(mesh['x']/1e3, mesh['y']/1e3, mesh['elements']-1)
pc = ax.tripcolor(mtri, 0*mesh['x'], facecolor='None', edgecolor='gray')
sc = ax.scatter(Xm, Ym, 10, rm, cmap=cmocean.cm.rain, vmin=0, vmax=5e-4)
ax.set_aspect('equal')
cbar = fig.colorbar(sc)
cbar.set_label('Runoff (kg m-2 s-1)')

# INTERPOLATE RUNOFF VALUES
pts = scipy.spatial.Delaunay(np.array([Xm, Ym]).T)
F = interpolate.LinearNDInterpolator(pts, rm, fill_value=0)
rinterp = F(np.array([mesh['x']/1e3, mesh['y']/1e3]).T)

fig,ax = plt.subplots(figsize=(8,4))

mtri = Triangulation(mesh['x']/1e3, mesh['y']/1e3, mesh['elements']-1)
pc = ax.tripcolor(mtri, rinterp, cmap=cmocean.cm.rain, edgecolor='none', vmin=0, vmax=5e-4)
sc = ax.scatter(Xm, Ym, 10, rm, cmap=cmocean.cm.rain, vmin=0, vmax=5e-4)
ax.set_aspect('equal')
cbar = fig.colorbar(sc)
cbar.set_label('Runoff (kg m-2 s-1)')



meshMelt = np.zeros((mesh['numberofvertices'], 1), dtype=np.float32)
F2 = interpolate.LinearNDInterpolator(pts, rall.T, fill_value=0)
rinterp = F2(np.array([mesh['x']/1e3, mesh['y']/1e3]).T).astype(np.float32)
print('rinterp:', rinterp.shape)

fig,ax = plt.subplots(figsize=(10, 4))
sortIndices = np.argsort(surface)
xx,yy = np.meshgrid(time[:tmax], surface[sortIndices])
pc = ax.pcolormesh(xx, yy, rinterp[sortIndices, :], cmap=cmocean.cm.rain)
ax.set_ylabel('Elevation (m asl.)')
# ax.set_xlabel('Day of year')
ax.set_ylim([np.min(surface), np.max(surface)])
cb = fig.colorbar(pc)
cb.set_label('Runoff (kg m-2 s-1)')
fig.subplots_adjust(left=0.08, right=0.995, bottom=0.1, top=0.95)
fig.savefig('RACMO_mesh_runoff.png', dpi=400)

print('rinterp.shape:', rinterp.shape)
out = np.zeros((rinterp.shape[0]+1, rinterp.shape[1]))
out[-1] = tyear[:tmax]
out[:-1] = rinterp

print(out[-1])

np.save('RACMO_mesh_runoff.npy', out)
