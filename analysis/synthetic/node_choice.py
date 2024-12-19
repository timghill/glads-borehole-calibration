import argparse
import numpy as np
import matplotlib
matplotlib.rc('font', size=8)
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib import patches
from utils.tools import import_config
import cmocean

colors = [
    cmocean.cm.ice(0.2),
    cmocean.cm.ice(0.5),
    cmocean.cm.ice(0.75),
]

lw = [1.5, 1.5, 1.5]
def node_choice(train_config, bh_config):
    train_config = import_config(train_config)

    mesh = np.load(train_config.mesh, allow_pickle=True)
    bh_config = np.load(bh_config, allow_pickle=True)
    dx = mesh['x'] - bh_config['x']
    dy = mesh['y'] - bh_config['y']
    D = np.sqrt(dx**2 + dy**2)
    node_sort = np.argsort(D)
    dist_sort = D[node_sort]
    print('Nearest nodes:', node_sort[:5])
    print('Distance:', dist_sort[:5])

    Y_sim = np.load(train_config.Y_physical, mmap_mode='r')

    fig,axs = plt.subplots(nrows=2, figsize=(5, 4), sharex=True)
    y = np.zeros((3, 365, 512))
    for j in range(3):
        y[j] = Y_sim[node_sort[j]*365+np.arange(365),:]
    ymu = np.mean(y, axis=(0,2))
    print('ymu', ymu.shape)
    yvar = np.var(y-ymu[None,:,None], axis=(0,1))
    print('yvar:', yvar.shape)
    qq = (0.95, 0.5)
    alphabet = ['(a)', '(b)']
    yvar_sortind = np.argsort(yvar)
    for i in range(len(qq)):
        yvar_ind = yvar_sortind[int(np.round(qq[i]*512))]
        print(yvar_sortind[yvar_ind])

        for j in range(3):
            axs[i].plot(np.arange(365), y[j, :, yvar_ind],
                color=colors[j], label='{:d} ({:.0f} m)'.format(node_sort[j], dist_sort[j]),
                linewidth=lw[j])

        ax = axs[i]
        ax.grid(linewidth=0.5)
        ax.set_xlim([120, 300])
        ax.set_ylabel('Flotation fraction')
        ax.text(0.025, 0.95, alphabet[i], fontweight='bold', transform=ax.transAxes,
            ha='left', va='top')
    axs[0].legend()
    
    axs[1].set_xlabel('Day of 2012')
    fig.subplots_adjust(bottom=0.1, left=0.12, right=0.95, top=0.95, hspace=0.1)
    fig.savefig('figures/node_choice.png', dpi=400)

    surf = np.load('../../issm/issm/data/geom/IS_surface.npy')
    bed = np.load('../../issm/issm/data/geom/IS_bed.npy')
    thick = surf-bed
    fig,ax = plt.subplots(figsize=(5,3))
    mtri = Triangulation(mesh['x']/1e3, mesh['y']/1e3, mesh['elements']-1)
    tripc = ax.tripcolor(mtri, bed, cmap=cmocean.cm.topo, vmin=-400, vmax=400,
        edgecolor='k', shading='flat')
    for j in range(3):
        xi = mesh['x'][node_sort[j]]/1e3
        yi = mesh['y'][node_sort[j]]/1e3
        ax.plot([xi,], [yi,], marker='s', color=colors[j],
        label='{:d} ({:.0f} m)'.format(node_sort[j], dist_sort[j]),
        linestyle='none',
        )
    ax.set_aspect('equal')
    ax.spines[['right', 'top']].set_visible(False)
    dx = 2.5e3
    dy = 1.5e3
    ax.set_xlim([bh_config['x']/1e3-dx/1e3, bh_config['x']/1e3+dx/1e3])
    ax.set_ylim([bh_config['y']/1e3-dy/1e3, bh_config['y']/1e3+dy/1e3])

    ax.set_xlabel('Easting (km)')
    ax.set_ylabel('Northing (km)')

    ax.plot(bh_config['x']/1e3, bh_config['y']/1e3, marker='^', color='b',
        markeredgecolor='w', label='GL12-2A', linestyle='none')
    ax.legend(bbox_to_anchor=(0,1.05,1,0.3), loc='lower center', ncols=2,
        frameon=False)
    fig.colorbar(tripc, label='Bed elevation (m asl.)')
    fig.subplots_adjust(left=0.17, bottom=0.1, right=0.95, top=0.8)
    fig.savefig('figures/node_map.png', dpi=400)

    connect = mesh['elements'] - 1
    connect_edge = mesh['connect_edge']
    
    vx = []
    vy = []
    ii = np.where(np.logical_and(np.abs(bh_config['x']/1e3 - mesh['x']/1e3)<=1.5*dx/1e3,
        np.abs(bh_config['y']/1e3 - mesh['y']/1e3)<=1.5*dy/1e3))[0]
    for i in ii:
        xi = mesh['x'][i]
        yi = mesh['y'][i]
        print(i)
        print(xi)
        print(yi)
        neigh_edges = np.where(np.any(connect_edge==i, axis=1))[0]
        neigh_nodes = connect_edge[neigh_edges][connect_edge[neigh_edges]!=i]
        vxi = 0.5*(xi + mesh['x'][neigh_nodes])
        vyi = 0.5*(yi + mesh['y'][neigh_nodes])
        if mesh['vertexonboundary'][i]:
            vxi = np.hstack((vxi, np.array([xi])))
            vyi = np.hstack((vyi, np.array([yi])))
        
        # Add elements
        neigh_els = np.where(np.any(connect==i, axis=1))[0]
        print("neigh_els:", neigh_els)
        neigh_els = neigh_els[neigh_els>=0]
        for k in range(len(neigh_els)):
            mpx = np.mean(mesh['x'][connect[neigh_els[k]]])
            mpy = np.mean(mesh['y'][connect[neigh_els[k]]])
            vxi = np.hstack((vxi, np.array([mpx])))
            vyi = np.hstack((vyi, np.array([mpy])))

        thetai = np.arctan2(vyi - yi, vxi - xi)
        argsort = np.argsort(thetai)
        vxi = vxi[argsort]
        vyi = vyi[argsort]

        vx.append(vxi)
        vy.append(vyi)


    fig,ax = plt.subplots(figsize=(5,3))
    norm = matplotlib.colors.Normalize(vmin=-400, vmax=400)
    print('ii:', ii)
    for i in range(len(ii)):
        ix = ii[i]
        # ax.plot(mesh['x'][ix]/1e3, mesh['y'][ix]/1e3, 'ko')
        xx = np.array((vx[i], vy[i])).T/1e3
        # print('xx:', xx.shape)
        poly = patches.Polygon(xx, alpha=1.0, edgecolor='none',
            facecolor=cmocean.cm.topo(norm(bed[ix])))
        ax.add_patch(poly)
    ax.set_xlim([bh_config['x']/1e3-dx/1e3, bh_config['x']/1e3+dx/1e3])
    ax.set_ylim([bh_config['y']/1e3-dy/1e3, bh_config['y']/1e3+dy/1e3])
    ax.set_aspect('equal')
    ax.tripcolor(mtri, 0*bed, facecolor='none', edgecolor='k')
    ax.set_xlabel('Easting (km)')
    ax.set_ylabel('Northing (km)')

    for j in range(3):
        xi = mesh['x'][node_sort[j]]/1e3
        yi = mesh['y'][node_sort[j]]/1e3
        ax.plot([xi,], [yi,], marker='s', color=colors[j],
        label='{:d} ({:.0f} m)'.format(node_sort[j], dist_sort[j]),
        linestyle='none',
        )
    ax.plot(bh_config['x']/1e3, bh_config['y']/1e3, marker='^', color='b',
        markeredgecolor='w', label='GL12-2A', linestyle='none')
    ax.legend(bbox_to_anchor=(0,1.05,1,0.3), loc='lower center', ncols=2,
        frameon=False)
    sm = matplotlib.cm.ScalarMappable(cmap=cmocean.cm.topo, norm=norm)
    fig.colorbar(sm, label='Bed elevation (m asl.)', ax=ax)
    fig.subplots_adjust(left=0.17, bottom=0.1, right=0.95, top=0.8)
    fig.savefig('figures/node_centred_map.png', dpi=400)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_config')
    parser.add_argument('bh_config')
    args = parser.parse_args()
    node_choice(args.train_config, args.bh_config)
