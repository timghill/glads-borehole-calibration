import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.tri import Triangulation
from scipy import interpolate
import rasterio
# import scipy.stats
# import scipy.optimize
# from osgeo import ogru
# import shapely
import shapefile
# import rasterio
import cmocean

#######################################
# Read moulin locations
#######################################

moulin_file = 'moulins_polarstereo.shp'
moulin_shp = shapefile.Reader(moulin_file)
XY = np.array([feat.shape.__geo_interface__['coordinates'] for feat in moulin_shp.shapeRecords()])
print(XY.shape)
fig,ax = plt.subplots()
ax.scatter(XY[:, 0], XY[:, 1], 5)
ax.set_aspect('equal')


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

fig,ax2 = plt.subplots(figsize=(12, 8))
elev = ax2.pcolormesh(dem_xx/1e3, dem_yy/1e3, dem, cmap='gray', vmin=0, vmax=2000)
ax2.contour(dem_xx/1e3, dem_yy/1e3, dem, levels=np.arange(100, 2000, 100), colors='k', linewidths=0.25, vmin=0, vmax=2000)
ax2.scatter(XY[:, 0]/1e3, XY[:, 1]/1e3, s=8, c='r', edgecolors='k')
ax2.set_aspect('equal')
ax2.set_xlabel('Easting (km)')
ax2.set_ylabel('Northing (km)')
ax2.yaxis.set_label_position('right')
ax2.yaxis.tick_right()


#######################################
# Read Mesh data
#######################################
mesh = np.load('../geom/IS_mesh.pkl', allow_pickle=True)
mtri = Triangulation(mesh['x']/1e3, mesh['y']/1e3, mesh['elements']-1)
ax2.tripcolor(mtri, 0*mesh['x'], facecolor='none', edgecolor='k', linewidth=0.5)
px = mesh['x'][mesh['elements']-1]
py = mesh['y'][mesh['elements']-1]
# Signed triangle area
S = 0.5*(-py[:,1]*px[:,2] + py[:,0]*(-px[:,1]+px[:,2]) + px[:,0]*(py[:,1] - py[:,2]) + px[:,1]*py[:,2])

inside_mx = []
inside_my = []
for (mx,my) in XY:
    # Compute barycentric coordinates
    s = 1/2/S * (py[:,0]*px[:,2] - px[:,0]*py[:,2] + (py[:,2]-py[:,0])*mx + (px[:,0]-px[:,2])*my)
    t = 1/2/S * (px[:,0]*py[:,1] - py[:,0]*px[:,1] + (py[:,0]-py[:,1])*mx + (px[:,1]-px[:,0])*my)
    in_triangles = (s>=0) & (t>=0) & ((1-s-t)>=0)
    # print('in_trangles:', in_triangles.shape)
    if np.any(in_triangles):
        ax2.plot(mx/1e3, my/1e3, 'b*')
        inside_mx.append(mx)
        inside_my.append(my)

n_inside = len(inside_mx)
print('Found {} moulins inside the domain'.format(n_inside))

# Interpolate moulins onto the mesh
dx = mesh['x'][:,None] - inside_mx
dy = mesh['y'][:,None] - inside_my
ds = np.sqrt(dx**2 + dy**2)
ds[mesh['vertexonboundary']==np.inf]
print('ds:', ds.shape)
inside_moulin_indices = np.argmin(ds, axis=0)
print('inside_moulin_indices:', inside_moulin_indices.shape)
moulin_indices, moulin_counts = np.unique(inside_moulin_indices, return_counts=True)
print('Number of unique moulins:', len(moulin_indices))
# print('Moulin counts:', moulin_counts)
counts_unique,counts_counts = np.unique(moulin_counts, return_counts=True)
print('Found nodes with repeat moulins:')
print(counts_unique)
print(counts_counts)

zmax = 1800
surf = np.load('../geom/IS_surface.npy')
xmax = np.max(mesh['x'][surf<=zmax])
xmin = np.min(mesh['x'][surf<=zmax])
ymax = np.max(mesh['y'][surf<=zmax])
ymin = np.min(mesh['y'][surf<=zmax])
ax2.set_xlim([xmin/1e3, xmax/1e3])
ax2.set_ylim([ymin/1e3, ymax/1e3])
bh_x = -205722.455632
bh_y = -2492714.204274


sc = ax2.scatter(mesh['x'][moulin_indices]/1e3, mesh['y'][moulin_indices]/1e3, 30, moulin_counts, vmin=1, vmax=4, cmap=cmocean.cm.thermal)
cb = fig.colorbar(sc)

ax2.plot(bh_x/1e3, bh_y/1e3, marker='^', color='magenta')

fig.subplots_adjust(left=0.05, bottom=0.05, top=0.95, right=0.95)
fig.savefig('moulins.png', dpi=400)

moulin_indices = np.sort(moulin_indices)
np.savetxt('moulin_indices_YS16.csv', moulin_indices, fmt='%d')

numberofelements = len(mesh['elements'])
# catchments = np.zeros(numberofelements, dtype=int)
xel = np.mean(mesh['x'][mesh['elements']-1], axis=1)
yel = np.mean(mesh['y'][mesh['elements']-1], axis=1)
zel = np.mean(surf[mesh['elements']-1], axis=1)
dx = xel[:,None] - mesh['x'][moulin_indices]
dy = yel[:,None] - mesh['y'][moulin_indices]
dz = zel[:,None] - surf[moulin_indices]
ds = np.sqrt(dx**2 + dy**2)
print('ds.shape:', ds.shape)

catchments = np.argmin(ds, axis=1)
catchments[zel<np.min(surf[moulin_indices])] = -1
print(catchments.min())
print(catchments.max())

fig,ax = plt.subplots(figsize=(12, 6))
ax.tripcolor(mtri, catchments)
ax.plot(mesh['x'][moulin_indices]/1e3, mesh['y'][moulin_indices]/1e3, 'ro', markersize=5)
ax.set_aspect('equal')
ax.set_xlim([xmin/1e3, xmax/1e3])
ax.set_ylim([ymin/1e3, ymax/1e3])
fig.savefig('catchments.png', dpi=400)

area_ele = mesh['area']

catchment_info = []
for i in range(len(moulin_indices)):
    cinfo = {}
    cinfo['area'] = np.sum(area_ele[catchments==i])/1e6
    cinfo['elements'] = np.where(catchments==i)[0]
    cinfo['moulin'] = moulin_indices[i]
    cinfo['area_units'] = 'km2'
    catchment_info.append(cinfo)

with open('moulins_catchments_YS16.pkl', 'wb') as outfile:
    pickle.dump(catchment_info, outfile)

# CHECKS
# plt.show()
