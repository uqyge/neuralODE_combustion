# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import cantera as ct
import pickle
from sklearn.metrics import r2_score
from src.dGenSensibleDecoupled import ignite_step
import plotly.express as px
#%%
columns, in_scaler, out_scaler = pickle.load(open('data/tmp.pkl', 'rb'))
species = columns
input_features = columns
labels = input_features

path = './of/ofMech'
# path = './of/init'
# path = './of/half'
# path = './of/noDiff'

df_of_c = pd.read_hdf(path + '/of_c.h5', key='of')
df_of_y = pd.read_hdf(path + '/of_Y.h5', key='of')

gas = ct.Solution('./data/connaire.cti')
gas_h0 = ct.Solution('./data/connaire.cti')
out_ode = []
out_c = []
for i in range(len(df_of_y)):
    if not (i % 1000):
        print(i)
    T = df_of_y['Temp'].values[i]
    Y_ini = df_of_y[gas.species_names].values[i]
    dt = df_of_y['dt'].values[i]
    a, b = ignite_step([T, Y_ini, 'H2', dt], gas, gas_h0)
    out_ode.append(b[0])
    out_c.append(a[0])
df_wdot_ode = pd.DataFrame(out_ode, columns=gas.species_names + ['Hs', 'Temp'])
df_c_ode = pd.DataFrame(out_c, columns=gas.species_names + ['Hs', 'Temp'])
df_c_ode['dt'] = 1

#%%
in_f_base = [
    'H', 'H2', 'O', 'O2', 'OH', 'H2O', 'N2', 'HO2', 'H2O2', 'Hs', 'Temp'
]
model_base = keras.models.load_model('./base_neuralODE_n64_b5_fcTrue.h5')
# df_of_c['Temp'] = df_of_c['T']
out_base = model_base.predict(in_scaler.transform(df_of_c[in_f_base]))
df_wdot_dnn_base = pd.DataFrame(out_scaler.inverse_transform(out_base),
                                columns=labels)

model = keras.models.load_model('./eulerModel.h5')
in_f = in_f_base + ['dt']
df_wdot_dnn = pd.DataFrame(model.predict(df_of_c[in_f]), columns=labels)
df_wdot_dnn_cmp = pd.DataFrame(model.predict(df_c_ode[in_f]), columns=labels)

plt.plot(df_wdot_dnn['Hs'], label='euler')
plt.plot(df_wdot_dnn_base['Hs'], label='base_net')
plt.plot(df_wdot_dnn_cmp['Hs'], label='compare')
plt.legend()
plt.title("network transform test")
plt.figure()

#%%
sp = 'Hs'
r2 = r2_score(df_of_c[sp], df_c_ode[sp])
plt.plot(df_of_c[sp], label="of")
plt.plot(df_c_ode[sp], label="ode")
plt.title("ode vs. of:{} r2={}".format(sp, r2))
plt.legend()
plt.show()

for sp in input_features:
    r2 = r2_score(df_of_c[sp], df_c_ode[sp])
    print("{}:r2={}".format(sp, r2))
#%%
# Compare ode and dnn
s = 50
e = 500
sp = 'H'
for sp in input_features:
    plt.subplot(1, 2, 1)
    plt.plot(df_wdot_ode[sp][s:e], label='ode')
    plt.plot(df_wdot_dnn[sp][s:e], label='dnn')
    # plt.plot(df_wdot_dnn_base[sp][s:e], label='dnn_base')
    # plt.plot(df_wdot_dnn_cmp[sp][s:e], label='dnn_cmp')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(df_of_c[sp][s:e])
    plt.title(sp)
    plt.figure()
    plt.show()
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
    dhdt.append(tmp)
df_wdot_dnn['Hs_calc'] = dhdt

s = 50
e = 350
# plt.plot(df_wdot_dnn['Hs_calc'][s:e], label='dnn_calc')
# plt.plot(df_wdot_dnn['Hs'][s:e], label='dnn')
plt.plot(df_wdot_ode['Hs'][s:e], label="ode")
plt.plot(df_of_y['RR_Hs'][s:e], label="of")
plt.title("Hs rate")
plt.legend()
plt.figure()
#%%
gas.TP = 1401, 10000
gas.X = 'H2:0.1,O2:1,N2:3.728'
print(gas['H2'].Y)
#%%
plt.plot(df_wdot_ode['H2O2'])
plt.show()

#%%
sp = 'H2O2'
for sp in input_features:
    vOf = df_of_c[sp].values[-1]
    vDb = t_org[sp].values[-1]
    print('{}:of={},db={},of/db={}'.format(sp, vOf, vDb, vOf / vDb))

#%%
# for sp in gas.species_names:
for sp in ['H2O2']:
    plt.plot(t_org['T'], t_org[sp], label='db')
    plt.plot(df_of_c['T'], df_of_c[sp], label='of')
    plt.legend()
    plt.title(sp)
    plt.show()

#%%

# for sp in input_features:
for sp in ['HO2']:
    plt.plot(df_of_c['T'], df_wdot_ode[sp], label='ode')
    plt.plot(df_of_c['T'], df_wdot_dnn[sp], label='dnn')
    plt.plot(t_org['T'], t_wdot[sp], label='db')
    plt.legend()
    plt.title(sp)
    plt.show()

#%%
a = (df_of_c['T'] < 1700) & (df_of_c['T'] > 1450)
print(sum(a))
#%%
plt.plot(df_of_c['T'], df_wdot_ode['H2O2'])
plt.plot(t_org['T'], t_wdot['H2O2'])

#%%
plt.plot(df_of_c['T'], df_of_c['H2O2'])
plt.plot(t_org['T'], t_org['H2O2'])
#%%
plt.plot(df_of_y['T'], df_of_y[gas.species_names].sum(1))

#%%
print(t_org['H2O2'].max(), df_of_c['H2O2'].max())

#%%
plt.plot(df_of_c['RR_Hs'])
plt.plot(df_wdot_dnn['Hs'])
#%%
