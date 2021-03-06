#%%
"""
This example creates batches of counterflow diffusion flame simulations at increasing strain rates.

The tutorial makes use of the scaling rules derived by Fiala and Sattelmayer
(doi:10.1155/2014/484372). Please refer to this publication for a detailed
explanation. Also, please don't forget to cite it if you make use of it.

This example can, for example, be used to iterate to a counterflow diffusion flame to an
awkward  pressure and strain rate, or to create the basis for a flamelet table.
"""
import copy
import multiprocessing as mp
import os
import shutil

import cantera as ct
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import tensorflow as tf
from molmass import Formula

# %%
# Create directory for output data files
data_directory = "diffusion_flame_batch_data/"
if not os.path.exists(data_directory):
    os.makedirs(data_directory)
if os.path.exists(data_directory):
    shutil.rmtree(data_directory)
    os.makedirs(data_directory)


# Define a limit for the maximum temperature below which the flame is
# considered as extinguished and the computation is aborted
# This increases the speed of refinement, if enabled
class FlameExtinguished(Exception):
    pass


temperature_limit_extinction = 900  # K


def interrupt_extinction(t):
    if np.max(f.T) < temperature_limit_extinction:
        raise FlameExtinguished("Flame extinguished")
    return 0.0


def Hs_T_rates(f):
    normalized_grid = f.grid / (f.grid[-1] - f.grid[0])
    T_org = copy.deepcopy(f.T)
    T_ref = 298.15

    hs_wdot = np.asarray(
        [
            np.dot(pm, -net)
            for pm, net in zip(f.partial_molar_enthalpies.T, f.net_production_rates.T)
        ]
    ).reshape(-1, 1)

    T_wdot = hs_wdot / (f.density * f.cp).reshape(-1, 1)

    Ha = np.asarray(
        [
            np.dot(pm, ha_Y / f.gas.molecular_weights)
            for pm, ha_Y in zip(f.partial_molar_enthalpies.T, f.Y.T)
        ]
    )

    f.set_profile("T", normalized_grid, f.T / f.T * T_ref)
    gas_ref = copy.deepcopy(f.partial_molar_enthalpies)

    H0 = np.asarray(
        [
            np.dot(pm, h0_Y / f.gas.molecular_weights)
            for pm, h0_Y in zip(gas_ref.T, f.Y.T)
        ]
    )

    # set temperature back
    f.set_profile("T", normalized_grid, T_org)
    grid = normalized_grid.reshape(-1, 1)

    return hs_wdot, T_wdot, (Ha - H0).reshape(-1, 1), grid


# INITIALIZATION
# reaction_mechanism = 'gri30.xml'
# reaction_mechanism = "./data/gri12/grimech12.cti"
reaction_mechanism = "./data/smooke.cti"
gas = ct.Solution(reaction_mechanism)
gas.transport_model = "UnityLewis"
width = 0.02  # 18mm wide
# width = 0.04  # 18mm wide
grid_ini = width * np.linspace(0, 1, 4000)
# grid_ini = width * (np.hstack([np.linspace(0, 0.9, 200), np.linspace(0.901, 1, 1800)]))

f = ct.CounterflowDiffusionFlame(gas, grid=grid_ini)
f.set_max_grid_points(f.flame, 20000)

# Define the operating pressure and boundary conditions
f.P = ct.one_atm  # 1 bar
# f.fuel_inlet.mdot = 0.25  # kg/m^2/s
f.fuel_inlet.mdot = 0.05  # kg/m^2/s

f.fuel_inlet.Y = "CH4:0.156364,O2:0.196486,N2:0.64715"
# f.fuel_inlet.X = 'CH4:0.00278,O2:0.00695,N2:0.02616'
f.fuel_inlet.T = 294  # K
# f.oxidizer_inlet.mdot = 0.5  # kg/m^2/s
f.oxidizer_inlet.mdot = 0.05  # kg/m^2/sy
f.oxidizer_inlet.Y = "O2:0.232917,N2:0.767083"
# f.oxidizer_inlet.X = 'O2:0.00695,N2:0.02616'
f.oxidizer_inlet.T = 294  # K

# Set refinement parameters, if used
# SMALL
# f.set_refine_criteria(ratio=100.0, slope=0.005, curve=0.01, prune=-0.1)
# Large
f.set_refine_criteria(ratio=100.0, slope=0.002, curve=0.002, prune=-0.1)
# XL
# f.set_refine_criteria(ratio=100.0, slope=0.001, curve=0.001, prune=-0.1)
# QUICK
# f.set_refine_criteria(ratio=100.0, slope=0.2, curve=0.5, prune=0.0001)

f.set_interrupt(interrupt_extinction)

# Initialize and solve
print("Creating the initial solution")
f.solve(loglevel=0, refine_grid=False, auto=True)

# Save to data directory
file_name = "initial_solution.xml"
f.save(
    data_directory + file_name,
    name="solution",
    description="Cantera version "
    + ct.__version__
    + ", reaction mechanism "
    + reaction_mechanism,
)


# STRAIN RATE LOOP
def flamelet_gen(i):
    # Compute counterflow diffusion flames at increasing strain rates at 1 bar
    # The strain rate is assumed to increase by 25% in each step until the flame is
    # extinguished
    # strain_factor = 1.2 ** i
    # 528
    # strain_factor = 1.009 ** i
    # 1200
    # strain_factor = 1.004 ** i
    # 300
    strain_factor = 75 * i[0] / (i[1] - 1) + 1

    exp_d_a = -1.0 / 2.0
    exp_u_a = 1.0 / 2.0
    exp_V_a = 1.0
    exp_lam_a = 2.0
    exp_mdot_a = 1.0 / 2.0

    # Restore initial solution
    file_name = "initial_solution.xml"
    f.restore(filename=data_directory + file_name, name="solution", loglevel=0)

    # Do the strain rate loop
    if np.max(f.T) > temperature_limit_extinction:
        print("strain rate iteration", i[0])
        # Create an initial guess based on the previous solution
        # Update grid
        f.flame.grid *= strain_factor ** exp_d_a
        normalized_grid = f.grid / (f.grid[-1] - f.grid[0])
        # Update mass fluxes
        f.fuel_inlet.mdot *= strain_factor ** exp_mdot_a
        f.oxidizer_inlet.mdot *= strain_factor ** exp_mdot_a
        # Update velocities
        f.set_profile("u", normalized_grid, f.u * strain_factor ** exp_u_a)
        f.set_profile("V", normalized_grid, f.V * strain_factor ** exp_V_a)
        # Update pressure curvature
        f.set_profile("lambda", normalized_grid, f.L * strain_factor ** exp_lam_a)
        try:
            # Try solving the flame
            f.solve(loglevel=0, refine_grid=True, auto=True)
            file_name = "strain_loop_" + format(i[0], "02d") + ".xml"
            f.save(
                data_directory + file_name,
                name="solution",
                loglevel=1,
                description="Cantera version "
                + ct.__version__
                + ", reaction mechanism "
                + reaction_mechanism,
            )
        except FlameExtinguished:
            print("Flame extinguished")

        except ct.CanteraError as e:
            print("Error occurred while solving:", e)

    strain_max = f.strain_rate("max")
    print(strain_max)

    return strain_max


# post processing
def read_flamelet(paraIn):
    file_name, col_names = paraIn
    print(file_name)

    f.restore(filename=data_directory + file_name, name="solution", loglevel=0)
    a_max = f.strain_rate("max")

    c = np.empty((0, len(col_names)), float)
    wdot = np.empty((0, len(col_names)), float)

    w_mat = f.net_production_rates
    c_mat = f.concentrations

    # c_mat = f.Y

    i = int(file_name.split(".")[0].split("_")[2])
    id_mat = np.ones((w_mat.shape[1], 1)) * i
    amax_mat = np.ones((w_mat.shape[1], 1)) * a_max
    Hs_w, T_w, Hs_c, grid = Hs_T_rates(f)
    T_c = f.T.reshape(-1, 1)

    tmp_c = np.hstack(
        (c_mat.T, Hs_c, T_c, f.density.reshape(-1, 1), id_mat, grid, amax_mat)
    )
    c = np.vstack((c, tmp_c))

    tmp_w = np.hstack(
        (w_mat.T, Hs_w, T_w, f.density.reshape(-1, 1), id_mat, grid, amax_mat)
    )
    wdot = np.vstack((wdot, tmp_w))

    return wdot, c


# %% parallel running
nRange = 1000
with mp.Pool() as pool:
    flamelet_range = [(x, nRange) for x in range(0, nRange, 1)]
    pool.map(flamelet_gen, flamelet_range)

# %% post processing
sp_names = f.gas.species_names
col_names = sp_names + ["Hs", "Temp", "rho", "id", "grid", "amax"]

#%%
n_files = os.listdir(data_directory)
n_files.remove("initial_solution.xml")

with mp.Pool() as pool:
    files = [(file_name, col_names) for file_name in n_files]
    raw = pool.map(read_flamelet, files)
wdot = np.vstack([out[0] for out in raw])
c = np.vstack([out[1] for out in raw])

print(f"There are {wdot.shape} samples.")

df_wdot = pd.DataFrame(wdot, columns=col_names)
df_c = pd.DataFrame(c, columns=col_names)

# set N2 to innert
df_wdot["N2"] = 0
df_wdot["AR"] = 0

out_name = "CH4_flt.h5"
if os.path.exists(out_name):
    os.remove(out_name)

df_c.to_hdf(out_name, key="c", format="table")
df_wdot.to_hdf(out_name, key="wdot", format="table")

# %% plot
px.scatter_3d(df_c.sample(frac=0.01), x="grid", y="amax", z="Temp", title="rate")

