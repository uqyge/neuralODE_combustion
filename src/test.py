# %%
import os
import numpy as np
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import tensorflow
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import optimizers

from src.dataScaling import data_scaler
from src.res_block import res_block
from src.utils import SGDRScheduler


# %%
dataPath = 'src/tmpHa.h5'
ddOld = dd.read_hdf(dataPath, key='old')
ddOrg = dd.read_hdf(dataPath, key='org')
ddNew = dd.read_hdf(dataPath, key='new')

ha_old = ddOld.compute()
ha_org = ddOrg.compute()
ha_new = ddNew.compute()

# %%
dataPath = 'src/tmpHs.h5'
ddOld = dd.read_hdf(dataPath, key='old')
ddOrg = dd.read_hdf(dataPath, key='org')
ddNew = dd.read_hdf(dataPath, key='new')

hs_old = ddOld.compute()
hs_org = ddOrg.compute()
hs_new = ddNew.compute()