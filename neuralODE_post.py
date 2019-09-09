# %%
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow.keras as keras
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.utils import plot_model

if os.path.exists('eulerModel.h5'):
    os.remove('eulerModel.h5')
if os.path.exists('rk4Model.h5'):
    os.remove('rk4Model.h5')

columns, in_scaler, out_scaler = pickle.load(open('data/tmp.pkl', 'rb'))
input_features = columns
labels = input_features

# %%
out_m = out_scaler.std.mean_.astype('float32')
out_s = out_scaler.std.scale_.astype('float32')

model_inv = Sequential(name='inv')
model_inv.add(
    Dense(len(out_m),
          input_dim=len(out_m),
          activation='linear',
          name='inv_out'))
# model_inv.add(Dense(len(out_m), input_dim=len(out_m), trainable=True))
# model_inv.add(Activation('linear'))
model_inv.layers[0].set_weights([(out_s) * np.identity(len(out_m)), +out_m])

in_m = in_scaler.std.mean_.astype('float32')
in_s = in_scaler.std.scale_.astype('float32')
# in_m = org[labels].mean().values
# in_s = org[labels].std().values

model_trans = Sequential(name='trans')
model_trans.add(Dense(len(in_m), input_dim=(len(in_m)), activation='linear'))
# model_trans.add(Dense(len(in_m), input_dim=(len(in_m)), trainable=True))
# model_trans.add(Activation('linear'))
model_trans.layers[0].set_weights([(1 / in_s) * np.identity(len(in_m)),
                                   -(in_m / in_s)])

# %%
# model_neuralODE = load_model('base_neuralODE_CH4_flt_n64_b5_fcTrue.h5')
model_neuralODE = load_model('base_neuralODE_CH4_sk_n64_b5_fcTrue.h5')
# model_neuralODE = load_model('base_neuralODE_CH4DB_n64_b5_fcTrue.h5')
# model_neuralODE = load_model('base_neuralODE_H2DB_n64_b5_fcTrue.h5')
model_neuralODE.summary()

# %%
post_model = Sequential(name='base')
post_model.add(model_trans)
post_model.add(model_neuralODE)
post_model.add(model_inv)
post_model.save('postODENet.h5')
post_model.summary()

# %%
dim_input = len(input_features)

in_0 = Input(shape=(dim_input + 1, ), name='input_0')
din = Dense(dim_input, activation='linear')(in_0)
k1 = post_model(din)

baseModel = Model(inputs=in_0, outputs=k1)
w_1 = np.vstack([np.identity(dim_input), np.zeros(dim_input)])
b_1 = np.zeros(dim_input)

baseModel.layers[1].set_weights([w_1, b_1])
baseModel.summary()

plot_model(baseModel, to_file="img/eulerModel.png")
baseModel.save('eulerModel.h5')

# %%
dim_input = len(input_features)

in_0 = Input(shape=(dim_input + 1, ), name='input_0')
# din = Input(shape=(dim_input, ), name='input_y')
# dt = Input(shape=(1, ), name='input_dt')

din = Dense(dim_input, activation='linear')(in_0)
dt = Dense(1, activation='linear')(in_0)

p1 = din
k1 = post_model(p1)

mul2 = keras.layers.multiply([k1, keras.layers.Lambda(lambda x: x * 0.5)(dt)])
p2 = keras.layers.add([mul2, p1])
k2 = post_model(p2)

mul3 = keras.layers.multiply([k2, keras.layers.Lambda(lambda x: x * 0.5)(dt)])
p3 = keras.layers.add([mul3, p1])
k3 = post_model(p3)

mul4 = keras.layers.multiply([k3, dt])
p4 = keras.layers.add([mul4, p1])
k4 = post_model(p4)

out1 = keras.layers.Lambda(lambda x: x * 1 / 6)(k1)
out2 = keras.layers.Lambda(lambda x: x * 1 / 3)(k2)
out3 = keras.layers.Lambda(lambda x: x * 1 / 3)(k3)
out4 = keras.layers.Lambda(lambda x: x * 1 / 6)(k4)
out = keras.layers.add([out1, out2, out3, out4], name='output')

# rk4Model = Model(inputs=[din, dt], outputs=out)
rk4Model = Model(inputs=in_0, outputs=out)
w_1 = np.vstack([np.identity(dim_input), np.zeros(dim_input)])
b_1 = np.zeros(dim_input)
w_2 = np.vstack([np.zeros((dim_input, 1)), np.ones(1)])
b_2 = np.zeros(1)
rk4Model.layers[1].set_weights([w_1, b_1])
rk4Model.layers[2].set_weights([w_2, b_2])

rk4Model.summary()

plot_model(rk4Model, to_file="fig/rk4Model.png")
rk4Model.save('rk4Model.h5')
