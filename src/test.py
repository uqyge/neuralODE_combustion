# %%
from scipy.integrate import odeint
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential, load_model
# from src.dataGen import test_data

import cantera as ct

from src.dataGenSensible import test_data
from src.dGenSensibleDecoupled import ignite_step

import tensorflow.keras as keras
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

#%%
columns, in_scaler, out_scaler = pickle.load(open('data/tmp.pkl', 'rb'))
# columns = org.columns
species = columns
input_features = columns.drop(['AR', 'dt', 'f', 'cp', 'Rho'])
labels = input_features

# %%
# path = './of/last'
path = './of/init'
# path = './of/init_divide'
# path = './of/noDiff'

data_c = path + '/of_c.h5'
data_y = path + '/of_Y.h5'
df_of_c = pd.read_hdf(data_c, key='of')
df_of_y = pd.read_hdf(data_y, key='of')

#%%
in_f_base = ['H', 'H2', 'O', 'O2', 'OH', 'H2O', 'N2', 'HO2', 'H2O2', 'Hs', 'T']
in_f_base_test = ['H', 'H2', 'O', 'O2', 'OH', 'H2O', 'HO2', 'H2O2', 'Hs']
model_base = keras.models.load_model('./base_neuralODE_n64_b5_fcTrue.h5')
out_base = model_base.predict(in_scaler.transform(df_of_c[in_f_base]))

df_wdot_dnn_base = pd.DataFrame(out_scaler.inverse_transform(out_base),
                                columns=labels)
#%%
model = keras.models.load_model('./eulerModel.h5')
in_f = in_f_base + ['dt']

# model = keras.models.load_model('./test.h5')
# in_f = in_f_base_test + ['dt']

out_model = model.predict(df_of_c[in_f])
df_wdot_dnn = pd.DataFrame(out_model, columns=labels)
# df_wdot_dnn = pd.DataFrame(out_model, columns=labels.drop(['T','N2']))
#%%
plt.plot(df_wdot_dnn['Hs'])
plt.plot(df_wdot_dnn_base['Hs'])
plt.figure()
#%%
gas = ct.Solution('./data/connaire.cti')
gas_h0 = ct.Solution('./data/connaire.cti')
out_ode = []
for i in range(len(df_of_y)):
    # for i in range(500):
    if not (i % 500):
        print(i)
    T = df_of_y['T'].values[i]
    Y_ini = df_of_y[gas.species_names].values[i]
    dt = df_of_y['dt'].values[i]
    a, b = ignite_step([T, Y_ini, 'H2', dt], gas, gas_h0)
    out_ode.append(b[0])
df_wdot_ode = pd.DataFrame(out_ode, columns=gas.species_names + ['Hs', 'T'])

#%%
# Compare ode and dnn
sp = 'H2O'
s = 30
e = 500
plt.subplot(1, 2, 1)
plt.plot(df_wdot_ode[sp][s:e], label='ode')
plt.plot(df_wdot_dnn[sp][s:e], label='dnn')
# plt.plot(df_wdot_dnn_base[sp][s:e], label='dnn_base')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(df_of_c[sp][s:e])
plt.title(sp)
plt.figure()
#%%
sp = 'Hs'
plt.subplot(1, 2, 1)
plt.plot(df_of_y[sp][s:e])
plt.subplot(1, 2, 2)
# plt.plot(df_of_y[sp][s:e])
plt.plot(df_wdot_dnn[sp][s:e])
plt.title("rr:,{}".format(sp))
plt.figure()
#%%
# Calculate dHdt
TRef = 298.15
gas.TP = TRef, ct.one_atm
df_wdot_dnn['AR'] = 0
df_wdot_dnn['N2'] = 0
dhdt = []
for i in range(len(df_wdot_dnn)):
    tmp = np.dot(gas.partial_molar_enthalpies,
                 -df_wdot_dnn[gas.species_names].values[i])
    # print(tmp)
    dhdt.append(tmp)
df_wdot_dnn['Hs_calc'] = dhdt

s = 50
e = 350
plt.plot(df_wdot_dnn['Hs_calc'][s:e], label='dnn_calc')
plt.plot(df_wdot_dnn['Hs'][s:e], label='dnn')
plt.plot(df_wdot_ode['Hs'][s:e], label="ode")
plt.plot(df_of_y['RR_Hs'][s:e], label="of")
plt.legend()
plt.figure()
#%%
gas.TP = 1401, 10000
gas.X = 'H2:0.1,O2:1,N2:3.728'
print(gas['H2'].Y)
