import numpy as np

reltol = np.logspace(-6, -4, 5)
dt = np.array([0.2, 0.5, 1, 2])[::-1]

tol,tt = np.meshgrid(reltol, dt)

print(tol.flatten(), tt.flatten())

njobs = len(tol.flatten())
jobarr = np.zeros((njobs, 3))
jobarr[:,0] = np.arange(1,njobs+1)
jobarr[:,1] = tol.flatten()
jobarr[:,2] = tt.flatten()

np.savetxt('table.dat', jobarr, fmt=('%d','%f','%f'))