from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Input
from keras.models import Model, Sequential
import tensorflow as tf
from keras import optimizers
import pandas as pd

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Activation
from keras.callbacks import ModelCheckpoint

from src.dataScaling import data_scaler
from src.res_block import res_block
from src.utils import SGDRScheduler

import pickle

df_load = pd.read_hdf('data/merged.h5')

df = df_load.astype('float32', copy=True)
# df=df_load
print(df.shape)

org = df[0:int(df.shape[0]/2)].reset_index(drop=True)
new = df[int(df.shape[0]/2):].reset_index(drop=True)
print(org.columns)

# %%
idx_dt = (org['dt'] < 5e-7) & (org['dt'] > 5e-9)
print(sum(idx_dt))
org = org[idx_dt]
new = new[idx_dt]

# %%
idx = (org > 1e-15).all(1)
print(sum(idx))
org = org[idx]
new = new[idx]

# %%
idx_f = ((new/org) < 5).all(1)
print(sum(idx_f))
org = org[idx_f]
new = new[idx_f]

# %%


def read_h5_data(input_features, labels):
    input_df = org[input_features]
    in_scaler = data_scaler()
    input_np = in_scaler.fit_transform(input_df[input_features].values, 'std2')

#     label_df=(new[labels]-org[labels])
    label_df = ((new[labels]-org[labels]).div(org.dt, axis=0))
#     label_df=(new[labels])/org[labels]
#     label_df=dydt[labels]

    out_scaler = data_scaler()
    label_np = out_scaler.fit_transform(label_df[labels].values, 'std2')

    return input_np, label_np, in_scaler, out_scaler


# %%
# define the labels
labels = df.columns.drop(['dt', 'f', 'Hs', 'cp'])
# labels=['HO2']

# input_features=df.columns.drop(['f'])
input_features = labels

# read in the data
x_input, y_label, in_scaler, out_scaler = read_h5_data(
    input_features=input_features, labels=labels)
x_train, x_test, y_train, y_test = train_test_split(
    x_input, y_label, test_size=0.9)
pickle.dump((org,new,in_scaler,out_scaler),open('./data/tmp.pkl','wb'))

# %%
print('set up ANN')

n_neuron = 10
scale = 3
branches = 3

# ANN parameters
dim_input = x_train.shape[1]
dim_label = y_train.shape[1]

batch_norm = False

# This returns a tensor
inputs = Input(shape=(dim_input,), name='input_1')

# x=Dense(len(in_m), trainable=False, activation='linear',
#         weights=[(1/in_s)*np.identity(len(in_m)),-(in_m/in_s)])(inputs)
# x = Dense(n_neuron, activation='relu')(x)

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(n_neuron, activation='relu')(inputs)

# less then 2 res_block, there will be variance
x = res_block(x, scale, n_neuron, stage=1, block='a',
              bn=batch_norm, branches=branches)
x = res_block(x, scale, n_neuron, stage=1, block='b',
              bn=batch_norm, branches=branches)
# x = res_block(x, scale, n_neuron, stage=1, block='c', bn=batch_norm,branches=branches)
# x = res_block(x, scale, n_neuron, stage=1, block='d', bn=batch_norm,branches=branches)
# x = res_block(x, scale, n_neuron, stage=1, block='e', bn=batch_norm,branches=branches)
# x = res_block(x, scale, n_neuron, stage=1, block='f', bn=batch_norm,branches=branches)


x = Dense(100, activation='relu')(x)
# x = Dropout(0.1)(x)
predictions = Dense(dim_label, activation='linear', name='output_1')(x)

baseModel = Model(inputs=inputs, outputs=predictions)

# out=Dense(len(out_m), trainable=False, activation='linear',
#           weights=[(out_s)*np.identity(len(out_m)),out_m] )(predictions)

# baseModel = Model(inputs=inputs, outputs=out)


# y=Dense(len(in_m), trainable=True,
#         weights=[(1/in_s)*np.identity(len(in_m)),-(in_m/in_s)])(inputs)
# y=baseModel(y)
# out=Dense(len(out_m), trainable=True,
#           weights=[(out_s)*np.identity(len(out_m)),out_m] )(y)
# model = Model(inputs=inputs,outputs=out)

model = baseModel
model.summary()

# %%

batch_size = 1024*8
epochs = 400
vsplit = 0.1

loss_type = 'mse'

model.compile(loss=loss_type, optimizer='adam', metrics=['accuracy'])
# model.compile(loss=loss_type, optimizer='adam', metrics=[coeff_r2])

# model.compile(loss=cubic_loss, optimizer=adam_op, metrics=['accuracy'])

# checkpoint (save the best model based validate loss)
!mkdir ./tmp
filepath = "./tmp/weights.best.cntk.hdf5"

checkpoint = ModelCheckpoint(filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min',
                             period=20)

epoch_size = x_train.shape[0]
a = 0
base = 2
clc = 2
for i in range(10):
    a += base*clc**(i)
print(a)
epochs, c_len = a, base
schedule = SGDRScheduler(min_lr=1e-5, max_lr=1e-3,
                         steps_per_epoch=np.ceil(epoch_size/batch_size),
                         cycle_length=c_len, lr_decay=0.8, mult_factor=2)

callbacks_list = [checkpoint]
# callbacks_list = [checkpoint, schedule]

model.load_weights(filepath)

# fit the model
history = model.fit(
    x_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=vsplit,
    verbose=2,
    callbacks=callbacks_list,
    shuffle=True)

model.save('base_neuralODE.h5')

#%%
from sklearn.metrics import r2_score
predict_val = model.predict(x_test,batch_size=1024*32)
predict_df = pd.DataFrame(out_scaler.inverse_transform(predict_val), columns=labels)
r2= r2_score(predict_val,y_test)
print(r2)