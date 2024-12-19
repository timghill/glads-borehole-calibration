"""
Read CSV file of boundary nodes and reformat to ISSM format
"""

import numpy as np

_header_template = """## Name:{fname}
## Icon:0
# Points Count  Value
{points} 1.000000
# X pos Y pos"""

fname_in = 'IS_outline.csv'
fname_out = 'IS_outline.exp'
hole_out = 'IS_hole.exp'


outline = np.loadtxt(fname_in, skiprows=1, quotechar='"', delimiter=',')
outline[:, 1:] = outline[:, 1:]

# For the true boundary
outline = outline[::20]
vertexid = outline[:, 0].astype(int)
n_vertices = len(vertexid)

reformat = np.zeros((n_vertices+1, 2))
reformat[:-1, :] = outline[:, 1:]
reformat[-1, :] = reformat[0, :]

header = _header_template.format(fname=fname_out, points=n_vertices+1)
np.savetxt(fname_out, reformat, header=header, delimiter=' ', comments='', fmt='%.2f')

# Now for refining the mesh we need to understand the approximate elevation of each outline vertex

