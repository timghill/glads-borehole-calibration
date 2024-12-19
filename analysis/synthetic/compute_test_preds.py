import argparse
import numpy as np
import scipy
import matplotlib
matplotlib.rc('font', size=8)
from matplotlib import pyplot as plt
from matplotlib import patches
import cmocean

from sepia.SepiaPredict import SepiaEmulatorPrediction

from utils import tools, models

def compute_test_preds(train_config, test_config, bh_config,
    nsamples=512, nburn=500, table='table.dat'):
    table = np.loadtxt(table, delimiter=' ', dtype=int)
    _,p,m = table[:].T
    bh_config = np.load(bh_config, allow_pickle=True)
    node = bh_config['node']
    y_ind_sim = node*365 + np.arange(365)
    model_days = np.arange(365)
    obs_days = np.arange(365)

    # keep track of overhead time to get to MCMC sampling

    # Load config files
    train_config = tools.import_config(train_config)
    test_config = tools.import_config(test_config)

    mcmap = cmocean.cm.deep
    mcolors = {
        128: mcmap(0.3),
        256: mcmap(0.5),
        512: mcmap(0.7),
        1024:mcmap(0.9),
    }

    mesh = np.load(train_config.mesh, allow_pickle=True)
    t_test = np.loadtxt(test_config.X_standard, delimiter=',', skiprows=1)
    Y_test = np.load(test_config.Y_physical).T

    for i in range(len(p)):
        # if p[i]==5:
        print('Working: ', m[i], p[i])
        train_config.p = p[i]
        train_config.m = m[i]
        # Read in the ensemble of simulations
        t_sim = np.loadtxt(train_config.X_standard, delimiter=',', skiprows=1)[:train_config.m].astype(np.float32)

        Y_sim_all = np.load(train_config.Y_physical, mmap_mode='r').T[:train_config.m, y_ind_sim]
        Y_sim = Y_sim_all[:, obs_days].astype(np.float32)

        # TESTING
        t_sim = t_sim[:Y_sim.shape[0]]

        print('Y_sim.shape', Y_sim.shape)

        # Find an appropriate candidate
        t_target = np.array([2./3., 0.5, 0.5, 0.5, 0.5, 0.8, 0.5, 3./4.])
        weights = np.array([1, 1, 0., 0., 0., 1, 0., 1])
        weighted_cost = np.sum(np.sqrt(weights**2 * (t_test - t_target)**2), axis=1)
        sim_num = np.argmin(weighted_cost)
        print('test sim number:', sim_num)
        
        Y_obs = Y_test[sim_num, y_ind_sim]
        t_true = t_test[sim_num]

        data,model = models.init_model_priors(y_sim=Y_sim, y_ind_sim=None, 
            y_obs=Y_obs[None,:], y_ind_obs=np.arange(Y_sim.shape[1]),
            t_sim=t_sim, p=train_config.p, t0=t_target)
        
        model_file = 'data/model_m{:03d}_p{:02d}'.format(m[i], p[i])
        model.restore_model_info(model_file)
        smp = model.get_samples(nsamples, nburn=nburn)
        preds = SepiaEmulatorPrediction(samples=smp, t_pred=t_test, model=model, addResidVar=True)

        ypred = preds.get_y()
        np.save('data/pred/y_test_pred_m{:03d}_p{:02d}.npy'.format(m[i], p[i]), ypred)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_config')
    parser.add_argument('test_config')
    parser.add_argument('bh_config')
    parser.add_argument('--sample', required=True, type=int)
    parser.add_argument('--burn', required=True, type=int)
    args = parser.parse_args()
    compute_test_preds(args.train_config, args.test_config, args.bh_config,
        nsamples=args.sample, nburn=args.burn)