#%%
import pandas as pd
import dask
import dask.dataframe as dd
from dask.delayed import delayed

import time
import cantera as ct
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

#%%
T_ref = 298.15
P = ct.one_atm

gas = ct.Solution('./data/connaire.cti')
gas_0 = ct.Solution('./data/connaire.cti')
# gas.TP = T_ref,P

test_y = np.array([
    0.00641602, 0.00485236, 8.66014e-05, 0.0691154, 5.32123e-06, 0.745117,
    0.153911, 0.00265207, 0.0178443, 1.44828e-07
])
test_T = 1361.52

gas.TPY = test_T, P, test_y
# gas_0.TPY = T_ref, P, test_y
gas_0.TP = T_ref, P
#%%
# h0 = gas.Y * gas.partial_molar_enthalpies / gas.molecular_weights
h0 = gas_0.partial_molar_enthalpies / gas_0.molecular_weights
print(gas_0.Y)
print('.......')
print(h0)

#%%
hc_dot = np.dot(h0, gas.net_production_rates)
print(hc_dot)
#%%
H2O_rate=gas['H2O'].net_production_rates*gas['H2O'].molecular_weights/gas.density
print(H2O_rate)
#%%
