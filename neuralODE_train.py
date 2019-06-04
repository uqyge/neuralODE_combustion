# %%
import os
import numpy as np
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import ModelCheckpoint

from src.dataScaling import data_scaler
from src.res_block import res_block
from src.utils import SGDRScheduler




# %%
dataPath = 'src/tmp.h5'
ddOld = dd.read_hdf(dataPath, key='old')
ddOrg = dd.read_hdf(dataPath, key='org')
ddNew = dd.read_hdf(dataPath, key='new')

idx_dt = (ddOld.dt < 5e-7) & (ddOld.dt > 5e-9)
ddOld = ddOld[idx_dt]
ddOrg = ddOrg[idx_dt]
ddNew = ddNew[idx_dt]
# ddOld.describe().compute()
#%%
# labels = ddOld.columns.drop(['AR','dt', 'f', 'cp','Hs'])
labels = ddOld.columns.drop(['AR','dt', 'f', 'cp'])
input_features = labels

idx = (ddOld[labels] > 0).all(1) & (ddOrg[labels] > 0).all(1) & (ddNew[labels] > 0).all(1)
ddOld = ddOld[idx]
ddOrg = ddOrg[idx]
ddNew = ddNew[idx]
#%%
# idx_f = abs((ddNew[labels]-ddOld[labels]).div(ddOld.dt +
#                                               ddOrg.dt, axis=0)).max(1) > 1000
# ddOld = ddOld[idx_f]
# ddOrg = ddOrg[idx_f]
# ddNew = ddNew[idx_f]

old = ddOld.compute()
org = ddOrg.compute()
new = ddNew.compute()

# logLab=['HO2','H2O2','O','OH','H2O','H']
# old[logLab] = np.log(old[logLab])
# org[logLab] = np.log(org[logLab])
# new[logLab] = np.log(new[logLab])
# old = old/old.max()
# org = org/org.max()
# new = new/new.max()
# old = np.cbrt(old)
# org = np.cbrt(org)
# new = np.cbrt(new)

# %%


def read_h5_data(input_features, labels):
    input_df = org[input_features]
    in_scaler = data_scaler()
    input_np = in_scaler.fit_transform(input_df[input_features].values, 'std2')

    label_df = ((new[labels]-old[labels]).div((org.dt+old.dt), axis=0))
    # label_df = ((new[labels]-org[labels]).div(org.dt, axis=0))
#     label_df=(new[labels])/org[labels]
#     label_df=dydt[labels]

    out_scaler = data_scaler()
    label_np = out_scaler.fit_transform(label_df[labels].values, 'std2')

    return input_np, label_np, in_scaler, out_scaler


# %%
# read in the data
x_input, y_label, in_scaler, out_scaler = read_h5_data(
    input_features=input_features, labels=labels)
x_train, x_test, y_train, y_test = train_test_split(
    x_input, y_label, test_size=0.05)
pickle.dump((org.columns, in_scaler, out_scaler), open('./data/tmp.pkl', 'wb'))

# %%
print('set up ANN')
# strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])
# with strategy.scope():
n_neuron = 100
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

model = baseModel
model.summary()
loss_type = 'mse'
model.compile(loss=loss_type, optimizer='adam', metrics=['accuracy'])


# %%
batch_size = 1024*8*8
epochs = 400
vsplit = 0.1
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
for i in range(7):
    a += base*clc**(i)
print(a)
epochs, c_len = a, base
schedule = SGDRScheduler(min_lr=1e-6, max_lr=1e-4,
                         steps_per_epoch=np.ceil(epoch_size/batch_size),
                         cycle_length=c_len, lr_decay=0.8, mult_factor=2)

callbacks_list1 = [checkpoint]
callbacks_list2 = [checkpoint, schedule]

model.load_weights(filepath)

# fit the model
history = model.fit(
    x_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=vsplit,
    verbose=2,
    callbacks=callbacks_list1,
    shuffle=True
)
# fit the model
history = model.fit(
    x_train, y_train,
    epochs=int(epochs/2),
    batch_size=batch_size,
    validation_split=vsplit,
    verbose=2,
    callbacks=callbacks_list2,
    shuffle=False
)
model.save('base_neuralODE.h5')

# %%
predict_val = model.predict(x_test, batch_size=1024*8*8*4)
predict_df = pd.DataFrame(
    out_scaler.inverse_transform(predict_val), columns=labels)
r2 = r2_score(predict_val, y_test)
print(r2)

# %%
pred_df = pd.DataFrame(predict_val, columns=labels)
test_df = pd.DataFrame(y_test, columns=labels)
spl_idx = pred_df.sample(frac=0.1).index
# sp='O'
for sp in labels:
    x = pred_df.sample(frac=0.1)
    plt.plot(pred_df.iloc[spl_idx][sp], test_df.iloc[spl_idx][sp], 'rd', ms=1)
    plt.title('{} r2 ={}'.format(sp, r2_score(pred_df[sp], test_df[sp])))
    plt.savefig('fig/{}_r2'.format(sp))
    plt.show()

# %%
sp = 'H2O2'
# grd = (np.log(new)-np.log(old)).div(old.dt+org.dt,axis=0)
grd = (new-old).div(old.dt+org.dt, axis=0)
# grd = new/old
plt.hist(grd[sp], bins=20)
plt.show()

# %%
