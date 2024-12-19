"""
Generate parameter experimental designs
"""

import sys
import os
import importlib
import argparse

import numpy as np
from scipy import stats

from matplotlib import pyplot as plt
import cmocean

def log_design(m, bounds, sampler=None):
    """
    Generate log-transformed parameter design.

    Note that bounds are specified in log space. For example,
    bounds = [0, 1] generates samples in physical space between
    1 and 10.

    Parameters
    ----------
    m : int
        Number of samples to draw

    bounds : (n_para, 2) array
             Lower and upper bounds on log parameters
    
    sampler : stats.qmc.QMCEngine, optional
              QMC sampler for generating the design. If not provided,
              defaults to stats.qmc.LatinHypercube
    
    Returns
    -------
    design : dict
             design['standard'] standardized design in [0, 1] hypercube
             design['log'] log design in provided bounds
             design['physical'] physical parameter values
    """
    if sampler is None:
        n_dim = bounds.shape[0]
        sampler = stats.qmc.LatinHypercube(n_dim, 
            optimization='random-cd', scramble=False, seed=42186)
    
    X_std = sampler.random(n=m)

    # Stretch [0, 1] interval into provided log bounds
    X_log = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0])*X_std

    # Convert to physical units
    X_phys = 10**X_log

    print('Generated', X_std.shape, 'design')

    print('Standardized min/max')
    print(X_std.min(axis=0), X_std.max(axis=0))

    print('Log min/max')
    print(X_log.min(axis=0), X_log.max(axis=0))

    print('Physical min/max')
    print(X_phys.min(axis=0), X_phys.max(axis=0))

    design = dict(standard=X_std, log=X_log, physical=X_phys)
    return design

def plot_design(design, bounds, para_names, figure=None):
    """
    Plot experimental design.

    Parwise plot of marginal histogram, scatter plot (lower triangular),
    and density plot (upper triangular).

    Parameters
    ----------
    design : dictionary
             experimental design computed using linear_design or log_design
    
    bounds : (n_para, 2) array
              Lower and upper bounds on parameters
    
    para_names : (n_para,) array of str
                  List of parameter names, used to label panels
    figure : optional, str or None
             Path to save figure
    
    Returns
    -------
    matplotlib.pyplot.figure
    """
    if 'log' in design.keys():
        scale = 'log'
        X = design['log']
    else:
        scale = 'linear'
        X = design['physical']
    
    dim = X.shape[1]

    fig, axs = plt.subplots(dim, dim, figsize=(6, 6))
    for ax in axs.flat:
        ax.set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    for i in range(dim):
        for j in range(dim):
            ax = axs[i,j]
            ax.set_visible(True)
            if j==i:
                ax.hist(X[:, i], density=True, log=False)
            elif j<i:
                ax.scatter(X[:, j], X[:, i], marker='x')
            elif j>i:
                vals = np.array([X[:, j], X[:, i]])
                kde = stats.gaussian_kde(np.array([X[:, j], X[:, i]]))
                x = np.linspace((bounds[j, 0]), (bounds[j, 1]), 101)
                y = np.linspace((bounds[i, 0]), (bounds[i, 1]), 101)
                [xx, yy] = np.meshgrid(x, y)
                positions = np.vstack([xx.ravel(), yy.ravel()])
                zz = np.reshape(kde(positions).T, xx.shape)
                ax.pcolormesh(xx, yy, zz, cmap=cmocean.cm.rain)
            if j>0:
                ax.set_yticklabels([])
            if i<(dim-1):
                ax.set_xticklabels([])
            if i==(dim-1):
                ax.set_xlabel(para_names[j])
            if j==0:
                ax.set_ylabel(para_names[i])
    if figure:
        fig.savefig(figure, dpi=400)
    return fig


def write_table(design, table_file='table.dat'):
    """Create table.dat for job array, compatible with Digital
    Research Alliance metafarm package.
    
    Parameters
    ----------
    design : dict
             Result of log_design
    
    table_file : str
                 File path to save table"""
    output_str = ''
    _line_template = '%d %d\n'

    m = design['physical'].shape[0]
    output_str = ''
    for i in range(1, m+1):
        output_str = output_str + _line_template % (i, i)
    output_str = output_str

    with open(table_file, 'w') as table_handle:
        table_handle.writelines(output_str)

def main():
    """
    Command-line interface to compute, plot and save experimental design

    python -m src.expdesign config

    where config is the path to a valid configuration file.
    """
    desc = 'Compute, plot and save experimental design'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('config', help='Path to experiment config file')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise OSError('Configuration file "{}" does not exist'.format(args.config))
    
    path, name = os.path.split(args.config)
    if path:
        abspath = os.path.abspath(path)
        sys.path.append(abspath)
    module, ext = os.path.splitext(name)
    config = importlib.import_module(module)

    design = log_design(config.m_max, config.theta_bounds,
        sampler=config.theta_sampler)
    para_fig = plot_design(design, config.theta_bounds,
        config.theta_names, config.exp + '.png')

    kwargs = dict(delimiter=',', fmt='%.6e',
        header=','.join(config.theta_names), comments='')
    np.savetxt(config.X_physical, design['physical'], **kwargs)
    np.savetxt(config.X_log, design['log'], **kwargs)
    np.savetxt(config.X_standard, design['standard'], **kwargs)

    write_table(design, table_file=config.table)
    return

if __name__=='__main__':
    main()