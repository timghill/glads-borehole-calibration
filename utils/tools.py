"""
Extra tools and utils for GP emulation project
"""

from pydoc import importfile

import time

import numpy as np
from scipy import stats

def import_config(conf_path):
    """
    Import a configuration file from path conf_path

    Parameters
    ----------
    conf_path : str
                Path to a valid python *.py file with GP model specification
    
    Returns
    -------
    module instance with configuration fields
    """
    return importfile(conf_path)

def reorder_edges(md):
    """
    Reorder edges from ISSM mesh md to make it easier to relate 
    channel discharge to the nodes of the mesh. Critical for plotting
    channel discharge and computing discharge across fluxgates.

    Based on plotchannels.m from ISSM model source.

    Parameters
    ----------
    md : model instance

    Returns:
    edges : (md.mesh.numberofedges, 2) array specifying
            ordered nodes connected to each edge
    """
    maxnbf = 3*mesh['numberofelements']
    edges = np.zeros((maxnbf, 3)).astype(int)
    exchange = np.zeros(maxnbf).astype(int)

    head_minv = -1*np.ones(md.mesh.numberofvertices).astype(int)
    next_face = np.zeros(maxnbf).astype(int)
    nbf = 0
    for i in range(mesh['numberofelements']):
        for j in range(3):
            v1 = mesh['elements'][i,j]-1
            if j==2:
                v2 = mesh['elements'][i,0]-1
            else:
                v2 = mesh['elements'][i,j+1]-1
            
            if v2<v1:
                v3 = v2
                v2 = v1
                v1 = v3
            
            exists = False
            e = head_minv[v1]
            while e!=-1:
                if edges[e, 1]==v2:
                    exists=True
                    break
                e = next_face[e]
            
            if not exists:
                edges[nbf,0] = v1
                edges[nbf,1] = v2
                edges[nbf,2] = i
                if v1!=mesh['elements'][i,j]-1:
                    exchange[nbf] = 1
                
                next_face[nbf] = head_minv[v1]
                head_minv[v1] = nbf
                nbf = nbf+1

    edges = edges[:nbf]
    pos = np.where(exchange==1)[0]
    v3 = edges[pos, 0]
    edges[pos, 0] = edges[pos, 1]
    edges[pos, 1] = v3

    edges = edges[:, :2]
    return edges


def reorder_edges_mesh(mesh):
    """
    Reorder edges from ISSM mesh md to make it easier to relate 
    channel discharge to the nodes of the mesh. Critical for plotting
    channel discharge and computing discharge across fluxgates.

    Based on plotchannels.m from ISSM model source.

    Parameters
    ----------
    md : model instance

    Returns:
    edges : (md.mesh.numberofedges, 2) array specifying
            ordered nodes connected to each edge
    """
    maxnbf = 3*mesh['numberofelements']
    edges = np.zeros((maxnbf, 3)).astype(int)
    exchange = np.zeros(maxnbf).astype(int)

    head_minv = -1*np.ones(mesh['numberofvertices']).astype(int)
    next_face = np.zeros(maxnbf).astype(int)
    nbf = 0
    for i in range(mesh['numberofelements']):
        for j in range(3):
            v1 = mesh['elements'][i,j]-1
            if j==2:
                v2 = mesh['elements'][i,0]-1
            else:
                v2 = mesh['elements'][i,j+1]-1
            
            if v2<v1:
                v3 = v2
                v2 = v1
                v1 = v3
            
            exists = False
            e = head_minv[v1]
            while e!=-1:
                if edges[e, 1]==v2:
                    exists=True
                    break
                e = next_face[e]
            
            if not exists:
                edges[nbf,0] = v1
                edges[nbf,1] = v2
                edges[nbf,2] = i
                if v1!=mesh['elements'][i,j]-1:
                    exchange[nbf] = 1
                
                next_face[nbf] = head_minv[v1]
                head_minv[v1] = nbf
                nbf = nbf+1

    edges = edges[:nbf]
    pos = np.where(exchange==1)[0]
    v3 = edges[pos, 0]
    edges[pos, 0] = edges[pos, 1]
    edges[pos, 1] = v3

    edges = edges[:, :2]
    return edges