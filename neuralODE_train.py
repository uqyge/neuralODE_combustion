#%%
import os
import pickle
import random
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

from src.ODENet import ODENetModel, SGDRScheduler

#%%
print("set up ANN")


def epoch_set(cycle=3):
    ep_size = 0
    base = 2
    clc = 2
    for i in range(cycle):
        ep_size += base * clc ** (i)
    print(ep_size)
    return base, ep_size


cycle = 10
c_len, epochs = epoch_set(cycle=cycle)

epoch_size = x_train.shape[0]
batch_size = 1024 * 8 * 8
vsplit = 0.1

scale = 3
fc = True
dataSet = dataPath.split("/")[1].split(".")[0]

for n_neuron in [64]:
    for branches in [5]:
        for fc in [True]:
            m_name = "{}_n{}_b{}_fc{}".format(dataSet, n_neuron, branches, fc)
            dim_input = x_train.shape[1]
            dim_label = y_train.shape[1]

            batch_norm = False

            model = ODENetModel(
                dim_input=dim_input,
                dim_label=dim_label,
                dataSet=dataSet,
                batch_norm=batch_norm,
                n_neuron=n_neuron,
                branches=branches,
                scale=scale,
                fc=fc,
            )
            model.summary()

            loss_type = "mse"
            model.compile(loss=loss_type, optimizer="adam", metrics=["accuracy"])

            filepath = "./tmp/{}.weights.best.cntk.hdf5".format(m_name)
            checkpoint = ModelCheckpoint(
                filepath,
                monitor="val_loss",
                verbose=1,
                save_best_only=True,
                mode="min",
                save_freq="epoch",
            )

            schedule = SGDRScheduler(
                min_lr=1e-6,
                max_lr=1e-4,
                steps_per_epoch=np.ceil(epoch_size / batch_size),
                cycle_length=c_len,
                lr_decay=0.8,
                mult_factor=2,
            )

            callbacks_list1 = [
                checkpoint,
                tensorflow.keras.callbacks.TensorBoard(
                    "./tb/{}".format(m_name), histogram_freq=0, profile_batch=0
                ),
            ]
            callbacks_list2 = callbacks_list1 + [schedule]

            # fit the model course
            # model.load_weights(filepath)
            history = model.fit(
                x_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=vsplit,
                verbose=2,
                callbacks=callbacks_list1,
                shuffle=False,
            )

            # fit the model refined
            history = model.fit(
                x_train,
                y_train,
                epochs=epoch_set(cycle - 1)[1],
                batch_size=batch_size,
                validation_split=vsplit,
                verbose=2,
                callbacks=callbacks_list2,
                shuffle=False,
            )
            model.save("base_neuralODE_{}.h5".format(m_name))

#%%
predict_val = model.predict(x_test, batch_size=1024 * 8 * 8 * 4)
predict_df = pd.DataFrame(out_scaler.inverse_transform(predict_val), columns=labels)
r2 = r2_score(predict_val, y_test)
print(r2)

# %%
pred_df = pd.DataFrame(predict_val, columns=labels)
test_df = pd.DataFrame(y_test, columns=labels)
spl_idx = pred_df.sample(frac=0.01).index

# for sp in labels:
for i in range(5):
    sp = random.choice(labels)
    x = pred_df.sample(frac=0.01)

    plt.subplot(2, 1, 1)
    plt.plot(pred_df.iloc[spl_idx][sp], test_df.iloc[spl_idx][sp], "rd", ms=1)
    plt.title("{} r2 ={}".format(sp, r2_score(pred_df[sp], test_df[sp])))
    plt.savefig("fig/{}_r2".format(sp))
    plt.subplot(2, 1, 2)
    plt.hist(wdot[sp], bins=20)
    plt.show()

    plt.savefig("fig/{}_r2".format(sp))


#%%
