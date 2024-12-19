import numpy as np
import datetime as dt

inputs = np.loadtxt('basin_integrated_inputs_RACMO.csv', delimiter=',')
ttdays = inputs[-1]

# ttdt = np.array([dt.datetime(2010,1,1) + dt.timedelta(days=ti) for ti in ttdays])

ttyears = 2010 + ttdays/365
print(len(ttyears))
print(len(np.unique(ttyears)))
print(ttyears)

inputs[-1] = ttyears
np.save('basin_integrated_inputs_RACMO.npy', inputs)
