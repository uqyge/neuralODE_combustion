# %%
import os
import pickle
import shutil

import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow
import tensorflow.keras as keras
from sklearn.metrics import r2_score
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.models import Model, Sequential

from src.dataScaling import data_scaler
from src.res_block import res_block
from src.utils import SGDRScheduler

# %%
print('set up ANN')
cycle = 6
n_neuron = 100
scale = 3
branches = 3
fc = True
dataSet = dataPath.split('/')[1].split('.')[0]

for n_neuron in [64]:
    for branches in [5]:
        for fc in [True]:
            m_name = '{}_n{}_b{}_fc{}'.format(dataSet, n_neuron, branches, fc)
            dim_input = x_train.shape[1]
            dim_label = y_train.shape[1]

            batch_norm = False

            # strategy = tensorflow.distribute.MirroredStrategy(devices=["/gpu:0"])
            # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

            # with strategy.scope():
            inputs = Input(shape=(dim_input, ), name='input_1')
            x = Dense(n_neuron, activation='relu')(inputs)

            # less then 2 res_block, there will be variance
            x = res_block(x,
                          scale,
                          n_neuron,
                          stage=1,
                          block='a',
                          bn=batch_norm,
                          branches=branches)
            x = res_block(x,
                          scale,
                          n_neuron,
                          stage=1,
                          block='b',
                          bn=batch_norm,
                          branches=branches)
            # x = res_block(x, scale, n_neuron, stage=1, block='c', bn=batch_norm,branches=branches)

            if fc == True:
                x = Dense(100, activation='relu')(x)
            # x = Dropout(0.1)(x)
            predictions = Dense(dim_label,
                                activation='linear',
                                name='output_1')(x)

            model = Model(inputs=inputs, outputs=predictions)
            model.summary()

            loss_type = 'mse'
            model.compile(loss=loss_type,
                          optimizer='adam',
                          metrics=['accuracy'])

            # %%
            print('Training')
            batch_size = 1024 * 8 * 8
            epochs = 400
            vsplit = 0.1

            # !mkdir ./tmp
            # if os.path.exists('./tmp'):
            # shutil.rmtree('./tmp')
            filepath = "./tmp/{}.weights.best.cntk.hdf5".format(m_name)

            checkpoint = ModelCheckpoint(filepath,
                                         monitor='val_loss',
                                         verbose=1,
                                         save_best_only=True,
                                         mode='min',
                                         save_freq='epoch')
            # period = 10)

            epoch_size = x_train.shape[0]
            ep_size = 0
            base = 2
            clc = 2
            for i in range(cycle):
                ep_size += base * clc**(i)
            print(ep_size)
            epochs, c_len = ep_size, base
            schedule = SGDRScheduler(min_lr=1e-6,
                                     max_lr=1e-4,
                                     steps_per_epoch=np.ceil(epoch_size /
                                                             batch_size),
                                     cycle_length=c_len,
                                     lr_decay=0.8,
                                     mult_factor=2)

            callbacks_list1 = [
                checkpoint,
                tensorflow.keras.callbacks.TensorBoard(
                    './tb/{}'.format(m_name),
                    histogram_freq=0,
                    profile_batch=0)
            ]
            callbacks_list2 = [
                checkpoint, schedule,
                tensorflow.keras.callbacks.TensorBoard(
                    './tb/{}'.format(m_name),
                    histogram_freq=0,
                    profile_batch=0)
            ]

            model.load_weights(filepath)

            # fit the model
            history = model.fit(x_train,
                                y_train,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_split=vsplit,
                                verbose=2,
                                callbacks=callbacks_list1,
                                shuffle=False)

            # fit the model
            history = model.fit(x_train,
                                y_train,
                                epochs=int(epochs / 2),
                                batch_size=batch_size,
                                validation_split=vsplit,
                                verbose=2,
                                callbacks=callbacks_list2,
                                shuffle=False)
            model.save('base_neuralODE_{}.h5'.format(m_name))

# %%
predict_val = model.predict(x_test, batch_size=1024 * 8 * 8 * 4)
predict_df = pd.DataFrame(out_scaler.inverse_transform(predict_val),
                          columns=labels)
r2 = r2_score(predict_val, y_test)
print(r2)

# %%
pred_df = pd.DataFrame(predict_val, columns=labels)
test_df = pd.DataFrame(y_test, columns=labels)
spl_idx = pred_df.sample(frac=0.01).index

# for sp in labels:
#     x = pred_df.sample(frac=0.01)
#     plt.plot(pred_df.iloc[spl_idx][sp], test_df.iloc[spl_idx][sp], 'rd', ms=1)
#     plt.title('{} r2 ={}'.format(sp, r2_score(pred_df[sp], test_df[sp])))
#     plt.savefig('fig/{}_r2'.format(sp))
#     plt.show()

# %%
sp = 'H2O2'
plt.hist(wdot[sp], bins=20)
plt.show()

#%%
