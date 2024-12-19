"""
Interpolate topography onto the triangular mesh
"""

import pickle
import numpy as np
from scipy import interpolate
from scipy import signal
import rasterio as rs

def interp_surf_bed(xy, 
    surface_file='BedMachineGreenland_StudyArea_surface.tif',
    bed_file='BedMachineGreenland_StudyArea_bed.tif'):

    surf_fid = rs.open(surface_file)
    bed_fid = rs.open(bed_file)

    surf = surf_fid.read(1)
    bed = bed_fid.read(1)

    xmin = surf_fid.bounds.left
    xmax = surf_fid.bounds.right
    ymin = surf_fid.bounds.bottom
    ymax = surf_fid.bounds.top
    nrows,ncols = surf_fid.shape

    surf_fid.close()
    bed_fid.close()

    x = np.linspace(xmin, xmax, ncols+1)[:-1]
    y = np.linspace(ymin, ymax, nrows+1)[0:-1][::-1]
    points = (x, y)

    surf_interp = interpolate.interpn(points, surf.T, xy, bounds_error=False, fill_value=-9999)
    bed_interp = interpolate.interpn(points, bed.T, xy, bounds_error=False, fill_value=-9999)

    # Extra step: compute surface gradient. We can 'cheat' by doing this on the regular grid
    # then interpolating
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dsurf_dx = np.zeros(surf.shape)
    dsurf_dy = np.zeros(surf.shape)
    dsurf_dx[:, 1:-1] = (surf[:, 2:] - surf[:, :-2])/2/dx
    dsurf_dy[1:-1,: ] = (surf[2:, :] - surf[:-2, :])/2/dy
    slope = np.sqrt(dsurf_dx**2 + dsurf_dy**2)

    # Smooth the slope
    size = 11
    window = np.ones((size, size))/size**2
    slope = signal.convolve2d(slope, window, mode='same')
    slope_interp = interpolate.interpn(points, slope.T, xy, bounds_error=False, fill_value=-9999)

    return surf_interp, bed_interp, slope_interp

if __name__=='__main__':
    with open('IS_mesh.pkl', 'rb') as meshin:
        mesh = pickle.load(meshin)
    xi = np.array([mesh['x'], mesh['y']]).T
    surf,bed,slope = interp_surf_bed(xi)
    np.save('IS_surface.npy', surf)
    np.save('IS_bed.npy', bed)
    np.save('IS_surface_slope.npy', slope)

    
