"""
Make ISSM meshes
"""
import os
import sys
import pickle
ISSM_DIR = os.getenv('ISSM_DIR')
sys.path.append(os.path.join(ISSM_DIR, 'bin/'))
sys.path.append(os.path.join(ISSM_DIR, 'lib/'))
from issmversion import issmversion
from model import model
from meshconvert import meshconvert
from solve import solve
from setmask import setmask
from parameterize import parameterize
from triangle import *
from bamg import *
from GetAreas import GetAreas
from plotmodel import *
import matplotlib
# matplotlib.use('QtAgg')
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.gridspec import GridSpec
import cmocean
# from save import *

from make_surface_bed import interp_surf_bed

from utils.tools import reorder_edges

## Point to domain outline file
outline = 'IS_outline.exp'
meshfile = 'IS_mesh.pkl'
min_length = 500
max_length = 5e3

min_elev = 1000
ELA_elev = 2000

## Mesh characteristics

## Make a draft mesh to interpolate surface elevations
md = model()
md = bamg(md, 'domain', 'IS_outline.exp', 'hmin', max_length, 'hmax', max_length, 'anisomax', 1.1)
print('Made draft mesh with numberofvertices:', md.mesh.numberofvertices)

## Elevation-dependent refinement
elev, bed, _ = interp_surf_bed(np.array([md.mesh.x, md.mesh.y]).T)
area = min_length + (max_length - min_length)*(elev-min_elev)/(ELA_elev-min_elev)
area[elev<min_elev] = min_length
area[elev>ELA_elev] = max_length

## Make the refined mesh
md = bamg(md, 'hVertices', area, 'anisomax', 1.1)
print('Refined mesh to have numberofvertices:', md.mesh.numberofvertices)

# Compute the nodes connected to each edge
connect_edge = reorder_edges(md)

# Compute edge lengths
x0 = md.mesh.x[connect_edge[:,0]]
x1 = md.mesh.x[connect_edge[:,1]]
dx = x1 - x0
y0 = md.mesh.y[connect_edge[:,0]]
y1 = md.mesh.y[connect_edge[:,1]]
dy = y1 - y0
edge_length = np.sqrt(dx**2 + dy**2)

if os.path.exists(meshfile):
    os.remove(meshfile)

meshdict = {}
meshdict['x'] = md.mesh.x
meshdict['y'] = md.mesh.y
meshdict['elements'] = md.mesh.elements
meshdict['area'] = GetAreas(md.mesh.elements, md.mesh.x, md.mesh.y)
meshdict['connect_edge'] = connect_edge
meshdict['edge_length'] = edge_length
meshdict['vertexonboundary'] = md.mesh.vertexonboundary
meshdict['numberofelements'] = md.mesh.numberofelements
meshdict['numberofvertices'] = md.mesh.numberofvertices
with open(meshfile, 'wb') as mesh:
    pickle.dump(meshdict, mesh)


