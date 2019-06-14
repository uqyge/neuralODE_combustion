# %%
from scipy.integrate import odeint
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Activation, Input
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.utils import plot_model
# %%
dim_input = 3
in_0 = Input(shape=(dim_input + 1, ), name='input_0')
x_dy = Dense(dim_input, activation='linear')(in_0)
x_dt = Dense(1, activation='linear')(in_0)

# w_1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
# w_2 = np.array([[0], [0], [0], [1]])

w_1 = np.vstack([np.identity(dim_input), np.zeros(dim_input)])
b_1 = np.zeros(dim_input)
w_2 = np.vstack([np.zeros((dim_input, 1)), np.ones(1)])
b_2 = np.zeros(1)
model = Model(inputs=in_0, outputs=[x_dy, x_dt])
model.layers[1].set_weights([w_1, b_1])
model.layers[2].set_weights([w_2, b_2])
model.summary()

# %%
t_0 = np.array([[10, 21, 33, 4]])
out = model.predict(t_0)
print(out)
#%%
