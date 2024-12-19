import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy import interpolate
import scipy.stats
import scipy.optimize
from osgeo import ogr
import rasterio

#######################################
# Read moulin locations
#######################################

moulin_file = 'moulins_polarstereo.shp'
moulin_shp = ogr.Open(moulin_file)
layer = moulin_shp.GetLayer(0)
n_moulins = layer.GetFeatureCount()
XY = np.zeros((n_moulins, 2))
for i, feature in enumerate(layer):
    gref = feature.GetGeometryRef()
    x, y = gref.GetPoint_2D()

    XY[i, 0] = x
    XY[i, 1] = y

#######################################
# Read ArcticDEM data
#######################################

arcticdem_file = 'arcticdem_mosaic_500m_v3-0_clipped.tif'
dem_img = rasterio.open(arcticdem_file)
dem = dem_img.read(1)
# print(dem_img.shape)
nr, nc = dem_img.shape
dem_xx = np.zeros((nr, nc))
dem_yy = np.zeros((nr, nc))
for ic in range(nc):
    for ir in range(nr):
        xy = dem_img.xy(ir, ic)
        dem_xx[ir, ic] = xy[0]
        dem_yy[ir, ic] = xy[1]

# fig, ax2 = plt.subplots(figsize=(4, 5))
fig = plt.figure(figsize=(7, 4))
gs = gridspec.GridSpec(2, 2, figure=fig,
    width_ratios=(1.5, 1), height_ratios=(1, 20),
    wspace=0, hspace=0.01,
    left=0.1, right=0.9, top=0.9)
ax2 = fig.add_subplot(gs[1, 1])
elev = ax2.pcolormesh(dem_xx/1e3, dem_yy/1e3, dem, cmap='gray', vmin=0, vmax=2000)
ax2.contour(dem_xx/1e3, dem_yy/1e3, dem, levels=np.arange(100, 2000, 100), colors='k', linewidths=0.25, vmin=0, vmax=2000)
ax2.scatter(XY[:, 0]/1e3, XY[:, 1]/1e3, s=8, c='r', edgecolors='k')
ax2.set_aspect('equal')
ax2.set_xlabel('Easting (km)')
ax2.set_ylabel('Northing (km)')
ax2.yaxis.set_label_position('right')
ax2.yaxis.tick_right()

# cax = fig.axes()
gs00 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, 1], width_ratios=(1, 8, 1))
cax = fig.add_subplot(gs00[1])
print(cax.get_position())
cbar = fig.colorbar(elev, cax=cax, orientation='horizontal', ticklocation='top')
cbar.set_label('Elevation (m asl)')
# cbar.ticklocation='top'


#######################################
# Interpolate elevation to moulin locations
#######################################
x_grid = dem_xx[0, :]
y_grid = dem_yy[:, 0]
y_grid = y_grid[::-1]
dem = dem[::-1, :]
print('X shape', x_grid.shape)
print('Y shape', y_grid.shape)
print('DEM shape', dem.shape)
grid_interpolant = interpolate.RegularGridInterpolator((y_grid, x_grid), dem, method='linear', bounds_error=False, fill_value=None)
moulin_elevation = grid_interpolant(XY[:, ::-1])


#######################################
# Moulin density
#######################################
dz = 100
zmin = dz*np.ceil(dem.min()/dz)
zmax = dz*np.floor(dem.max()/dz)
z = np.arange(zmin, zmax, dz)
nz = len(z)
dx = x_grid[1] - x_grid[0]
dy = y_grid[1] - y_grid[0]
cell_area = dz*dy
density = np.zeros(nz)
for i in range(nz):
    z1 = z[i]
    z2 = z1 + dz

    moulin_mask = moulin_elevation[np.logical_and(moulin_elevation>=z1, moulin_elevation<z2)]
    cells = dem[np.logical_and(dem>=z1, dem<z2)]

    n_moulin = len(moulin_mask)
    n_cell = len(cells)
    area = n_cell*dx*dy/1e6

    interval_density = n_moulin/area
    density[i] = interval_density
    # print(interval_density)

save_arr = np.zeros((len(z), 2))
save_arr[:, 0] = z
save_arr[:, 1] = density
print(save_arr)
np.savetxt('yang2016_density.txt', save_arr)

elevation_bins = np.zeros(len(z))
n_moulin_bins = np.zeros(len(z))
for i in range(nz):
    z1 = z[i]
    z2 = z1 + dz
    mask = np.logical_and(moulin_elevation>=z1, moulin_elevation<z2)
    # cells = dem[mask]
    elevation_bins[i] = z1
    n_moulin_bins[i] = len(moulin_elevation[mask])




zz = (elevation_bins*n_moulin_bins)/sum(n_moulin_bins)
# zz = zz[zz>0]
# print(elevation_bins*n_moulin_bins)
mean_elev = np.mean(len(elevation_bins)*zz)

var = np.average( (elevation_bins - mean_elev)**2, weights=n_moulin_bins)
std_elev = np.sqrt(var)

def gauss(x, A, x0, sigma):
    return A*scipy.stats.norm.pdf(x, loc=x0, scale=sigma)
p, v, idf, mesg, ier = scipy.optimize.curve_fit(gauss, z, density, full_output=True, p0=[np.max(density), mean_elev, std_elev])

def chi(x, A, x0, sigma):
    k=5
    return A * scipy.stats.chi.pdf((x-x0)/sigma, k)
p_chi, v_chi = scipy.optimize.curve_fit(chi, z, density, p0=[np.max(density), mean_elev, std_elev])

def tdist(x, A, x0, sigma):
    v = 2
    return A * scipy.stats.t.pdf((x-x0)/sigma, v)
p_t, v_t = scipy.optimize.curve_fit(tdist, z, density, p0=[np.max(density), mean_elev, std_elev])

print('Gauss curve_fit:', p)
print('Chi curve fit:', p_chi)
print('t curve fit:', p_t)


# std_elev = np.std(all_density)

# fig4, ax4 = plt.subplots()
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(z, density, label='Data')
# ax4.hist(moulin_elevation)
# ax4.axvline(mean_elev, color='k')
# ax4.fill_betweenx([0, 0.06], [mean_elev-std_elev, mean_elev-std_elev],
#     [mean_elev+std_elev, mean_elev+std_elev], alpha=0.2)
# ax4.plot(z, 50*scipy.stats.norm.pdf(z, loc=mean_elev, scale=std_elev))
ax4.plot(z, gauss(z, *p), label='Normal')
# ax4.plot(z, chi(z, *p_chi), label='Chi')
# ax4.plot(z, tdist(z, *p_t), label='t')
ax4.grid()
ax4.set_xlabel('Elevation (m asl)')
ax4.set_ylabel('Moulin density (km$^{-2}$)')
ax4.legend()

fig.savefig('yang2016_moulin_density_normalfit.png', dpi=600)
plt.show()
