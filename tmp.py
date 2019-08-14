#%%
from src.ODENet import ODENetModel, SGDRScheduler
import tensorflow
from tensorflow.keras.callbacks import ModelCheckpoint

# from src.utils import SGDRScheduler

#%%
model = ODENetModel(dim_input=x_train.shape[1], dim_label=y_train.shape[1])

loss_type = "mse"
model.compile(loss=loss_type, optimizer="adam", metrics=["accuracy"])
model.summary()
#%%
cycle = 3

print("Training")
batch_size = 1024 * 8 * 8
epochs = 400
vsplit = 0.1

m_name = "wudi"
filepath = "./tmp/{}.weights.best.cntk.hdf5".format(m_name)

checkpoint = ModelCheckpoint(
    filepath,
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    mode="min",
    save_freq="epoch",
)

epoch_size = x_train.shape[0]
ep_size = 0
base = 2
clc = 2
for i in range(cycle):
    ep_size += base * clc ** (i)
print(ep_size)
epochs, c_len = ep_size, base
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

# fit the model
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

#%%
