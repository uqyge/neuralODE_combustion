# %%
import os
import numpy as np
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from src.dataScaling import data_scaler

# %%
dataPath = 'src/H2DB.h5'
ddOrg = dd.read_hdf(dataPath, key='c')
ddWdot = dd.read_hdf(dataPath, key='wdot')

# %%
input_features = [
    'H', 'H2', 'O', 'O2', 'OH', 'H2O', 'N2', 'HO2', 'H2O2', 'Hs', 'Temp'
]
labels = input_features

org = ddOrg.compute()
wdot = ddWdot.compute()


# %%
def read_h5_data(input_features, labels):
    in_scaler = data_scaler()
    out_scaler = data_scaler()
    input_df = org[input_features]
    input_np = in_scaler.fit_transform(input_df[input_features].values, 'std2')

    # label_df = ((new[labels]-old[labels]).div((org.dt+old.dt), axis=0))
    label_df = wdot[labels]
    label_np = out_scaler.fit_transform(label_df[labels].values, 'std2')

    return input_np, label_np, in_scaler, out_scaler


x_input, y_label, in_scaler, out_scaler = read_h5_data(
    input_features=input_features, labels=labels)
x_train, x_test, y_train, y_test = train_test_split(x_input,
                                                    y_label,
                                                    test_size=0.05)
pickle.dump((labels, in_scaler, out_scaler), open('./data/tmp.pkl', 'wb'))

#%%
