"""
Interpolate topography onto the triangular mesh
"""

import pickle
import numpy as np
from scipy import interpolate
from scipy import signal
import rasterio as rs

def interp_surf_bed(xy, 
    vel_file='GrIMP_multiyear_vel_mosaic_vv.tif'):

    vel_fid = rs.open(vel_file)

    vv = vel_fid.read(1)

    xmin = vel_fid.bounds.left
    xmax = vel_fid.bounds.right
    ymin = vel_fid.bounds.bottom
    ymax = vel_fid.bounds.top
    nrows,ncols = vel_fid.shape

    vel_fid.close()

    x = np.linspace(xmin, xmax, ncols+1)[:-1]
    y = np.linspace(ymin, ymax, nrows+1)[0:-1][::-1]

    points = (x, y)

    vv_interp = interpolate.interpn(points, vv.T, xy, bounds_error=False, fill_value=-9999)

    return vv_interp

if __name__=='__main__':
    with open('../geom/IS_mesh.pkl', 'rb') as meshin:
        mesh = pickle.load(meshin)
    xi = np.array([mesh['x'], mesh['y']]).T
    vv = interp_surf_bed(xi)
    np.save('IS_surface_velocity.npy', vv)
    
