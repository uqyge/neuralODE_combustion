# %%
from scipy.integrate import odeint
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential, load_model
from src.dataGen import test_data

org, _, in_scaler, out_scaler = pickle.load(open('data/tmp.pkl', 'rb'))
columns = org.columns
species = org.columns
labels = org.columns.drop(['dt', 'f', 'Hs', 'cp'])
input_features = labels

# %%
out_m = out_scaler.std.mean_.astype('float32')
out_s = out_scaler.std.scale_.astype('float32')
# out_m = (new[labels]-org[labels]).div(org.dt,axis=0).mean().values
# out_s = (new[labels]-org[labels]).div(org.dt,axis=0).std().values


model_inv = Sequential()
model_inv.add(Dense(len(out_m), input_dim=len(out_m), trainable=True))
model_inv.add(Activation('linear'))
model_inv.layers[0].set_weights([(out_s)*np.identity(len(out_m)), +out_m])

in_m = in_scaler.std.mean_.astype('float32')
in_s = in_scaler.std.scale_.astype('float32')
# in_m = org[labels].mean().values
# in_s = org[labels].std().values

model_trans = Sequential()
model_trans.add(Dense(len(in_m), input_dim=(len(in_m)), trainable=True))
model_trans.add(Activation('linear'))
model_trans.layers[0].set_weights(
    [(1/in_s)*np.identity(len(in_m)), -(in_m/in_s)])

# %%
model_neuralODE = load_model('base_neuralODE.h5')
model_neuralODE.summary()

# %%
post_model = Sequential()
post_model.add(model_trans)
post_model.add(model_neuralODE)
post_model.add(model_inv)


# %%
post_model.predict(org[labels].iloc[0:1])

out_scaler.inverse_transform(model_neuralODE.predict(
    in_scaler.transform(org[labels].iloc[0:1])))


# %%


def euler(data_in, dt):

    pred = data_in[input_features]

    #     print(i)
    # model_pred = pd.DataFrame(out_scaler.inverse_transform(model_neuralODE.predict(
    #     in_scaler.transform(pred), batch_size=1024*8)), columns=labels)
    model_pred = pd.DataFrame(post_model.predict(
        pred, batch_size=1024*8), columns=labels)

    pred = (model_pred)*dt + pred

    return pred


def rk2(data_in, dt):

    pred = data_in[input_features]

    k1 = pd.DataFrame(post_model.predict(
        pred, batch_size=1024*8), columns=labels)
    mid = k1*(dt/2)+pred

    k2 = pd.DataFrame(post_model.predict(
        mid, batch_size=1024*8), columns=labels)

    pred = k2 * dt + pred

    return pred


def rk4(data_in, dt):
    p1 = data_in
    k1 = pd.DataFrame(post_model.predict(
        p1[input_features], batch_size=1024*8), columns=labels)
    # print(k1)
    p2 = k1*dt/2+data_in
    k2 = pd.DataFrame(post_model.predict(
        p2[input_features], batch_size=1024*8), columns=labels)

    p3 = k2*dt/2 + data_in
    k3 = pd.DataFrame(post_model.predict(
        p3[input_features], batch_size=1024*8), columns=labels)

    p4 = k3*dt + data_in
    k4 = pd.DataFrame(post_model.predict(
        p4[input_features], batch_size=1024*8), columns=labels)

    model_pred = 1/6*(k1+2*k2+2*k3+k4)
    pred = data_in + model_pred*dt
    # print(model_red)
    return pred


def nvAd(data_in, dt, odeAg, st):
    # st = 10
    pred = data_in[input_features]

    for i in range(st):
        pred = odeAg(pred, dt/st)

    return pred, (pred-data_in[input_features])/dt


# %%

def dydt(x, t):
    out = post_model.predict(x.reshape(1, -1))
    return out.flatten()


def odeInt(data_in, dt):
    out_ode = []
    for i in range(len(data_in)):
        x0 = data_in[input_features].iloc[i:i+1]
        ode_out = odeint(
            dydt,
            x0.values[0],
            [0, dt]
        )
        out_ode.append(ode_out[1].reshape(1, -1))

    out = np.concatenate(out_ode)
    out = pd.DataFrame(out, columns=labels)

    return out, (out-data_in[labels])/dt


solvers = {'euler': euler,
           'midpoint': rk2,
           'rk4': rk4
           }
# %%
# post_species = species.drop(['cp', 'Hs', 'Rho','dt','f','N2'])
post_species = pd.Index(['HO2', 'OH', 'H2O2', 'H2'])

st = 1
ini_T = 1401
dt = 1e-6
solver = 'euler'
for n in [2]:
    input_0, test = test_data(ini_T, n, columns, dt)

    input_0 = input_0.reset_index(drop=True)
    test = test.reset_index(drop=True)

    pred, model_pred = nvAd(input_0, dt, solvers[solver], st)
    # pred, model_pred = odeInt(input_0, dt)

    test_target = ((test-input_0) / dt)

    testGrad = pd.DataFrame(out_scaler.transform(
        test_target[labels]), columns=labels)
    trGrad = pd.DataFrame(out_scaler.transform(
        model_pred[labels]), columns=labels)

    for sp in post_species.intersection(species):
        f, axarr = plt.subplots(1, 3)
        f.suptitle('{}: {}, T={}'.format(solver.upper(), sp, ini_T))

        axarr[0].plot(test[sp])
        axarr[0].plot(pred[sp], 'rd', ms=2)
        # axarr[0].set_title(str(n) + '_' + sp)

#       axarr[1].plot((test[sp] - pred[sp]) / test[sp], 'k')
        axarr[1].plot((test[sp] - pred[sp])/test[sp].max(), 'y')

#         axarr[1].set_ylim(-0.005, 0.005)
        # axarr[1].set_title(str(n) + '_' + sp)

        # ax2 = axarr[1].twinx()
        # ax2.plot(test_target[sp], 'bd', ms=2)
        # # ax2.plot(model_pred[sp], 'rd', ms=2)
#       ax2.set_ylim(-0.0015,0.0015)

        axarr[2].plot(testGrad[sp], 'bd', ms=2)
        axarr[2].plot(trGrad[sp], 'rd', ms=2)
#       axarr[2].set_ylim(-0.1,0.)

#           ax2.plot(no_scaler[sp], 'md', ms=2)

        plt.savefig('fig/' + '{}_{}_{}_{}'.format(st, solver, ini_T, sp))
        plt.show()
