# %%
import os
import pickle

import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.ODENet import data_scaler

import sys

sys.path.append("/home/edison/repos/ofegplots")
from ofegplots import etl, euler_pred, plot_compare, plot_single, read_of
from ofegplots import ct_chem
import cantera as ct
from molmass import Formula


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


# %% Extract data
# dataPath = "data/CH4_sk_S_P1000.h5"
dataPath = "./CH4_flt.h5"
# dataPath = "./CH4_flt_t.h5"

ddOrg = dd.read_hdf(dataPath, key="c")
ddWdot = dd.read_hdf(dataPath, key="wdot")
org = ddOrg.compute()
wdot = ddWdot.compute()
org = org.clip(0)

gas = ct.Solution("./data/smooke.cti")
species = gas.species_names

*input_species, _ = species
# input_features = input_species + ["Hs", "Temp"]
input_features = input_species + ["Temp"]

labels = input_features

df_Y = pd.DataFrame()
for sp in gas.species_names:
    df_Y[sp] = org[sp] * Formula(sp).mass / org.rho
df_Y["Temp"] = org["Temp"]
df_Y[["grid", "amax"]] = org[["grid", "amax"]]
df_Y["f"] = (df_Y["N2"].max() - df_Y["N2"]) / (df_Y.N2.max() - df_Y.N2.min())

org["f"] = df_Y["f"]
wdot["f"] = df_Y["f"]

px.scatter(
    org[(org.amax > 10) & (org.amax < 2000)].sample(frac=0.2),
    x="f",
    y="Temp",
    color="amax",
)

#%%
amax = list(set(org.amax))
amax.sort()
plt.plot(amax)


# %%
model = tf.keras.models.load_model("eulerModel.h5")
org["dt"] = 1e-8
df_dnn = pd.DataFrame(
    model.predict(org[input_features + ["dt"]], batch_size=1024 * 8),
    columns=input_features,
)

df_dnn["grid"] = org["grid"]
df_dnn["amax"] = org["amax"]
df_dnn["f"] = df_Y["f"]


# %%
# frac = 0.1
# sp = "H2O"
# px.scatter_3d(org.sample(frac=frac), x="grid", y="amax", z=sp)
#%%
# px.scatter_3d(wdot.sample(frac=frac), x="grid", y="amax", z=sp)
#%%
# px.scatter_3d(df_dnn.sample(frac=frac), x="grid", y="amax", z=sp)


#%%
px.line(wdot[wdot["amax"] < 50], x="grid", y="H", color="amax")


#%%
sp = "O"
px.scatter_3d(df_Y.sample(frac=frac), x="f", y="amax", z=sp)

#%%
sp = "O"
px.scatter_3d(
    df_Y[((org.f < 0.5) & (org.f > 0.45))].sample(frac=frac), x="f", y="amax", z=sp
)

#%%
T = 1933.1337890625
Y = np.array(
    [
        1.12565067e-02,
        2.15311162e-03,
        3.22684828e-05,
        1.01490039e-03,
        7.09635469e-06,
        9.66265574e-02,
        3.86503860e-02,
        1.62130478e-03,
        6.41749575e-05,
        1.93062183e-02,
        1.37524214e-04,
        1.11260230e-03,
        6.69718975e-06,
        1.15355611e-01,
        1.50356840e-07,
        7.12655067e-01,
    ]
)

#%%
gas.Y = Y
gas.TP = T, ct.one_atm
gas.net_production_rates
#%%
gas["H2O"].net_production_rates

#%%
idx = wdot.H2O.argmax()

#%%
df_Y.iloc[idx, :][gas.species_names]

#%%
px.scatter(
    wdot[(wdot.amax > 10) & (wdot.amax < 1000)].sample(frac=0.2),
    x="f",
    y="O2",
    color="amax",
)

#%%
px.scatter(
    df_dnn[(df_dnn.amax > 100) & (df_dnn.amax < 1800)].sample(frac=0.2),
    x="f",
    y="H2",
    color="amax",
)

#%%
px.scatter(
    org[(org.amax > 100) & (org.amax < 1800)].sample(frac=0.2),
    x="f",
    y="Temp",
    color="amax",
)

#%%
px.scatter(org[(org.amax < 100)].sample(frac=0.2), x="f", y="Temp", color="amax")


#%%
plt.hist(org.f[(org.amax > 1700)], bins=20)

#%%
plt.hist(org.f[(org.amax < 100)], bins=20)

#%%
