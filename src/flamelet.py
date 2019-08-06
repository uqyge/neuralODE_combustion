#%%
# This file is part of Cantera. See License.txt in the top-level directory or
# at http://www.cantera.org/license.txt for license and copyright information.
"""
This example creates two batches of counterflow diffusion flame simulations.
The first batch computes counterflow flames at increasing pressure, the second
at increasing strain rates.

The tutorial makes use of the scaling rules derived by Fiala and Sattelmayer
(doi:10.1155/2014/484372). Please refer to this publication for a detailed
explanation. Also, please don't forget to cite it if you make use of it.

This example can, for example, be used to iterate to a counterflow diffusion flame to an
awkward  pressure and strain rate, or to create the basis for a flamelet table.
"""

import copy
import os
import shutil

import cantera as ct
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import tensorflow as tf
from molmass import Formula

#%%
# Create directory for output data files
data_directory = 'diffusion_flame_batch_data/'
if os.path.exists(data_directory):
    shutil.rmtree(data_directory)
    os.makedirs(data_directory)

if not os.path.exists(data_directory):
    os.makedirs(data_directory)


# Define a limit for the maximum temperature below which the flame is
# considered as extinguished and the computation is aborted
# This increases the speed of refinement, if enabled
class FlameExtinguished(Exception):
    pass


temperature_limit_extinction = 900  # K


def interrupt_extinction(t):
    if np.max(f.T) < temperature_limit_extinction:
        raise FlameExtinguished('Flame extinguished')
    return 0.


def Hs_T_rates(f):
    normalized_grid = f.grid / (f.grid[-1] - f.grid[0])
    T_org = copy.deepcopy(f.T)
    T_ref = 298.15

    hs_wdot = np.asarray([
        np.dot(pm, -net) for pm, net in zip(f.partial_molar_enthalpies.T,
                                            f.net_production_rates.T)
    ]).reshape(-1, 1)

    T_wdot = hs_wdot / (f.density * f.cp).reshape(-1, 1)

    Ha = np.asarray([
        np.dot(pm, ha_Y / f.gas.molecular_weights)
        for pm, ha_Y in zip(f.partial_molar_enthalpies.T, f.Y.T)
    ])

    f.set_profile('T', normalized_grid, f.T / f.T * T_ref)
    gas_ref = copy.deepcopy(f.partial_molar_enthalpies)

    H0 = np.asarray([
        np.dot(pm, h0_Y / f.gas.molecular_weights)
        for pm, h0_Y in zip(gas_ref.T, f.Y.T)
    ])

    # set temperature back
    f.set_profile('T', normalized_grid, T_org)
    grid = normalized_grid.reshape(-1, 1)

    return hs_wdot, T_wdot, (Ha - H0).reshape(-1, 1), grid


#%% PART 1: INITIALIZATION

# reaction_mechanism = 'h2o2.xml'
# reaction_mechanism = 'gri30.xml'
reaction_mechanism = './data/gri12/grimech12.cti'
gas = ct.Solution(reaction_mechanism)
width = 0.02  # 18mm wide
grid_ini = width * np.linspace(0, 1, 20)
# f = ct.CounterflowDiffusionFlame(gas, width=width)

f = ct.CounterflowDiffusionFlame(gas, grid=grid_ini)
f.set_max_grid_points(f.flame, 10000)

# Define the operating pressure and boundary conditions
f.P = ct.one_atm  # 1 bar
# f.fuel_inlet.mdot = 0.25  # kg/m^2/s
f.fuel_inlet.mdot = 0.05  # kg/m^2/s
# f.fuel_inlet.X = 'CH4:0.00278,O2:0.00695,N2:0.02616'
f.fuel_inlet.Y = 'CH4:1'
f.fuel_inlet.T = 300  # K
# f.oxidizer_inlet.mdot = 0.5  # kg/m^2/s
f.oxidizer_inlet.mdot = 0.1  # kg/m^2/s
f.oxidizer_inlet.Y = 'O2:0.23,N2:0.77'
# f.oxidizer_inlet.X = 'O2:0.00695,N2:0.02616'
f.oxidizer_inlet.T = 300  # K

# Set refinement parameters, if used
# f.set_refine_criteria(ratio=100.0, slope=0.002, curve=0.005, prune=0.0001)
f.set_refine_criteria(ratio=100.0, slope=0.2, curve=0.5, prune=0.0001)

f.set_interrupt(interrupt_extinction)

# Initialize and solve
print('Creating the initial solution')
f.solve(loglevel=0, refine_grid=False, auto=True)

# Save to data directory
file_name = 'initial_solution.xml'
f.save(data_directory + file_name,
       name='solution',
       description='Cantera version ' + ct.__version__ +
       ', reaction mechanism ' + reaction_mechanism)

#%% STRAIN RATE LOOP

# Compute counterflow diffusion flames at increasing strain rates at 1 bar
# The strain rate is assumed to increase by 25% in each step until the flame is
# extinguished
# strain_factor = 1.0201
strain_factor = 1.21

# Exponents for the initial solution variation with changes in strain rate
# Taken from Fiala and Sattelmayer (2014)
exp_d_a = -1. / 2.
exp_u_a = 1. / 2.
exp_V_a = 1.
exp_lam_a = 2.
exp_mdot_a = 1. / 2.

# Restore initial solution
file_name = 'initial_solution.xml'
f.restore(filename=data_directory + file_name, name='solution', loglevel=0)

# Counter to identify the loop
n = 0
# Do the strain rate loop
while np.max(f.T) > temperature_limit_extinction:
    n += 1
    print('strain rate iteration', n)
    # Create an initial guess based on the previous solution
    # Update grid
    f.flame.grid *= strain_factor**exp_d_a
    normalized_grid = f.grid / (f.grid[-1] - f.grid[0])
    # Update mass fluxes
    f.fuel_inlet.mdot *= strain_factor**exp_mdot_a
    f.oxidizer_inlet.mdot *= strain_factor**exp_mdot_a
    # Update velocities
    f.set_profile('u', normalized_grid, f.u * strain_factor**exp_u_a)
    f.set_profile('V', normalized_grid, f.V * strain_factor**exp_V_a)
    # Update pressure curvature
    f.set_profile('lambda', normalized_grid, f.L * strain_factor**exp_lam_a)
    try:
        # Try solving the flame
        f.solve(loglevel=0, refine_grid=True)
        file_name = 'strain_loop_' + format(n, '02d') + '.xml'
        f.save(data_directory + file_name,
               name='solution',
               loglevel=1,
               description='Cantera version ' + ct.__version__ +
               ', reaction mechanism ' + reaction_mechanism)
    except FlameExtinguished:
        print('Flame extinguished')
        break
    except ct.CanteraError as e:
        print('Error occurred while solving:', e)
        break

    print(f.strain_rate('max'))
    if (n == 2):
        break

#%%
sp_names = f.gas.species_names
col_names = sp_names + ['Hs'] + ['Temp'] + ['id'] + ['grid'] + ['amax']
c = np.empty((0, len(col_names)), float)
wdot = np.empty((0, len(col_names)), float)

for i in range(n - 1):
    print(i)
    file_name = 'strain_loop_{0:02d}.xml'.format(i + 1)
    f.restore(filename=data_directory + file_name, name='solution', loglevel=0)
    a_max = f.strain_rate('max')

    w_mat = f.net_production_rates
    c_mat = f.concentrations
    id_mat = np.ones((w_mat.shape[1], 1)) * i
    amax_mat = np.ones((w_mat.shape[1], 1)) * a_max
    Hs_w, T_w, Hs_c, grid = Hs_T_rates(f)

    tmp_c = np.hstack((c_mat.T, Hs_c, f.T.reshape(-1,
                                                  1), id_mat, grid, amax_mat))
    c = np.vstack((c, tmp_c))

    tmp_w = np.hstack((w_mat.T, Hs_w, T_w, id_mat, grid, amax_mat))
    wdot = np.vstack((wdot, tmp_w))

df_wdot = pd.DataFrame(wdot, columns=col_names)
df_c = pd.DataFrame(c, columns=col_names)

df_c.to_hdf('CH4_flt.h5', key='c', format='table')
df_wdot.to_hdf('CH4_flt.h5', key='wdot', format='table')

#%%
px.scatter_3d(df_c.sample(frac=1),
              x='grid',
              y='id',
              z='Temp',
              title='concentration')
# #%%
# px.scatter_3d(df_wdot.sample(frac=0.001), x='grid', y='id', z='N2', title='rate')
# # %%
# print(n)
# id_slt = 11
# y_last = df_wdot.Hs[df_wdot.id == id_slt]
# x_last = df_c.Temp[df_c.id == id_slt]
# plt.plot(x_last, y_last)

#%%
input_features = [
    "H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2", "C", "CH", "CH2",
    "CH2(S)", "CH3", "CH4", "CO", "CO2", "HCO", "CH2O", "CH2OH", "CH3O",
    "CH3OH", "C2H", "C2H2", "C2H3", "C2H4", "C2H5", "C2H6", "HCCO", "CH2CO",
    "HCCOH", "N2", 'Hs', 'Temp'
]
# df_c=org
# df_wdot=wdot

df_c['dt'] = 1e-8

model = tf.keras.models.load_model('eulerModel.h5')
pred = model.predict(df_c[input_features + ['dt']], batch_size=1024 * 8)
df_dnn = pd.DataFrame(pred, columns=input_features)

df_dnn['grid'] = df_c['grid']
df_dnn['id'] = df_c['id']
#%%
px.scatter_3d(df_dnn.sample(frac=0.01), x='grid', y='id', z='OH', title='dnn')

#%%
