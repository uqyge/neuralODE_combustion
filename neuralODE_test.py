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


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


# %% Extract data
# dataPath = 'data/CH4DB.h5'
# dataPath = 'data/H2DB_L.h5'
# dataPath = "data/CH4_flt.h5"
dataPath = "data/CH4_sk.h5"

ddOrg = dd.read_hdf(dataPath, key="c")
ddWdot = dd.read_hdf(dataPath, key="wdot")

# species = [
#     'H', 'H2', 'O', 'O2', 'OH', 'H2O', 'N2', 'HO2', 'H2O2'
# ]
species = [
    "CH4", "CH3", "CH3O", "CH2O", "HCO", "CO2", "CO", "H2", "H", "O2", "O",
    "OH", "HO2", "H2O", "H2O2", "N2"
]
# species = [
#     "H2",
#     "H",
#     "O",
#     "O2",
#     "OH",
#     "H2O",
#     "HO2",
#     "H2O2",
#     "C",
#     "CH",
#     "CH2",
#     "CH2(S)",
#     "CH3",
#     "CH4",
#     "CO",
#     "CO2",
#     "HCO",
#     "CH2O",
#     "CH2OH",
#     "CH3O",
#     "CH3OH",
#     "C2H",
#     "C2H2",
#     "C2H3",
#     "C2H4",
#     "C2H5",
#     "C2H6",
#     "HCCO",
#     "CH2CO",
#     "HCCOH",
#     "N2",
#     "AR",
# ]

*input_species, _ = species
# input_features = input_species + ["Hs", "Temp"]
input_features = input_species + ["Temp"]

labels = input_features

org = ddOrg.compute()
wdot = ddWdot.compute()

# %%

# %%
model = tf.keras.models.load_model("eulerModel.h5")
org["dt"] = 1e-8
df_dnn = pd.DataFrame(
    model.predict(org[input_features + ["dt"]], batch_size=1024 * 8),
    columns=input_features,
)

df_dnn["grid"] = org["grid"]
df_dnn["amax"] = org["amax"]

# %%
frac = 0.01
sp = "H2O"

px.scatter_3d(org.sample(frac=frac), x="grid", y="amax", z=sp)
#%%
px.scatter_3d(df_dnn.sample(frac=frac), x="grid", y="amax", z=sp)

#%%
px.scatter_3d(wdot.sample(frac=frac), x="grid", y="amax", z=sp)

#%%
px.line(wdot[wdot["amax"]<50], x="grid", y="O",color="amax")

#%%
px.line(org[wdot["amax"]<50], x="grid", y="O",color="amax")

#%%
