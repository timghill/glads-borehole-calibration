"""
Collect outputs from individual simulations into .npy arrays.

Command line interface

    python -m src.aggregate_outputs issm [config] [njobs] [--all]
"""

import os
import sys
import importlib
import argparse
import pickle

import numpy as np

from . import tools

def collect_issm_results(config, njobs, dtype=np.float32, save_all=False):
    """
    Collect outputs from individual simulations into .npy arrays.

    Parameters
    ----------
    config : module
             Imported configuration file
    
    njobs : int
            Number of jobs to look for in the results directory
    
    dtype : type
            Data type to cast outputs into. Recommend np.float32
    
    save_all : bool
               If save_all is True, save values for all scalar
               variable definitions in addition to the mean.
               If save_all is False, save values for only
               the mean of each variable across definitions.
    """
    with open(config.mesh, 'rb') as meshin:
        mesh = pickle.load(meshin)
    
    nodes = np.array([mesh['x'], mesh['y']]).T
    connect = mesh['elements'].astype(int)-1
    connect_edge = mesh['connect_edge'].astype(int)
    edge_length = mesh['edge_length']

    # Construct file patterns
    jobids = np.arange(1, njobs+1)

    resdir = 'RUN/output_{:03d}/'
    respattern = os.path.join(resdir, '{}.npy')
    aggpattern = '{exp}_{}.npy'.format('{}', exp=config.exp)
    testout = np.load(respattern.format(1, 'ff'))
    nt = 365
    all_ff = np.zeros((mesh['numberofvertices']*nt, njobs), dtype=dtype)
    all_S = np.zeros((len(edge_length)*nt, njobs), dtype=dtype)
    all_Q = np.zeros((len(edge_length)*nt, njobs), dtype=dtype)
    all_hs = np.zeros((mesh['numberofvertices']*nt, njobs), dtype=dtype)
    for i,jobid in enumerate(jobids):
        print('Job %d' % jobid)
        ff = np.load(respattern.format(jobid, 'ff'))[:, -nt-1:-1]
        all_ff[:, i] = ff.flatten()

        Q = np.load(respattern.format((jobid), 'Q'))[:, -nt-1:-1]
        S = np.load(respattern.format((jobid), 'S'))[:, -nt-1:-1]
        h_s = np.load(respattern.format((jobid), 'h_s'))[:, -nt-1:-1]
        all_Q[:, i] = Q.flatten()
        all_S[:, i] = S.flatten()
        all_hs[:, i] = h_s.flatten()
        
    np.save(aggpattern.format('ff'), all_ff)
    np.save(aggpattern.format('S'), all_S)
    np.save(aggpattern.format('Q'), all_Q)
    np.save(aggpattern.format('hs'), all_hs)
    return 

def main():
    """
    Command-line interface to collect simulation outputs
    """
    desc = 'Collect simulation outputs'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('config', help='Path to experiment config file')
    parser.add_argument('njobs', help='Number of jobs', type=int)
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise OSError('Configuration file "{}" does not exist'.format(args.config))
    
    config = tools.import_config(args.config)
    collect_issm_results(config, args.njobs)

if __name__=='__main__':
    main()
