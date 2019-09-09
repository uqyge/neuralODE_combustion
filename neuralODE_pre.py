# %%
import os
import pickle

import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.ODENet import data_scaler

# %% Extract data
# dataPath = 'data/CH4DB.h5'
# dataPath = 'data/H2DB_L.h5'
# dataPath = "data/CH4_flt.h5"
# dataPath = "./CH4_flt.h5"
dataPath = "data/CH4_sk.h5"

ddOrg = dd.read_hdf(dataPath, key="c")
ddWdot = dd.read_hdf(dataPath, key="wdot")

# %%
# species = [
#     'H', 'H2', 'O', 'O2', 'OH', 'H2O', 'N2', 'HO2', 'H2O2'
# ]
species = [
    "CH4",
    "CH3",
    "CH3O",
    "CH2O",
    "HCO",
    "CO2",
    "CO",
    "H2",
    "H",
    "O2",
    "O",
    "OH",
    "HO2",
    "H2O",
    "H2O2",
    "N2",
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


# %% prepare data for training
def read_h5_data(input_features, labels):
    in_scaler = data_scaler()
    out_scaler = data_scaler()
    input_df = org[input_features]
    input_np = in_scaler.fit_transform(input_df[input_features].values, "std2")

    # label_df = ((new[labels]-old[labels]).div((org.dt+old.dt), axis=0))
    label_df = wdot[labels]
    label_np = out_scaler.fit_transform(label_df[labels].values, "std2")

    return input_np, label_np, in_scaler, out_scaler


x_input, y_label, in_scaler, out_scaler = read_h5_data(
    input_features=input_features, labels=labels
)
x_train, x_test, y_train, y_test = train_test_split(x_input, y_label, test_size=0.05)
pickle.dump((labels, in_scaler, out_scaler), open("./data/tmp.pkl", "wb"))
