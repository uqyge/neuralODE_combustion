#%%
import numpy as np
import matplotlib.pyplot as plt
# import scipy.integrate
from scipy.integrate import solve_ivp


#%%
def f1(t, y):
    dydt = y * y - y * y * y
    return dydt


delta = 0.01
steps = 50000
t = np.linspace(0, 2 / delta, steps)
t_end = 2 / delta

#%%
# md = 'RK45'
md = 'BDF'
sol = solve_ivp(
    f1,
    (0, 2 / delta),
    [delta],
    t_eval=t,
    # method='RK45',
    method=md,
    # max_step=delta,
    rtol=1e-5)

plt.plot(sol.t, sol.y[0], 'rd')
plt.title('steps:{}'.format(sol.y.shape[1]))
plt.show()

#%%
new = sol.y[0][1:-1]
org = sol.y[0][0:-2]
grd = (new - org) / ((2 / delta) / (steps - 1))

plt.plot(grd)
# plt.ylim(-0.3,-0.1)
plt.ylim(-5e-4, 5e-4)
plt.title('gradient')
plt.savefig('./fig/flame_gradient_{}'.format(md))
plt.show()
#%%
plt.plot(grd)
from sklearn.preprocessing import StandardScaler
inS = StandardScaler()
outS = StandardScaler()
x_train = inS.fit_transform(org.reshape(-1, 1))
y_train = outS.fit_transform(grd.reshape(-1, 1))

#%%
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers


def baseline_model():
    model = Sequential()
    model.add(Dense(200, 'relu', input_dim=1))
    model.add(Dense(800, 'relu'))
    model.add(Dense(800, 'relu'))
    model.add(Dense(1, activation='linear'))

    return model


base = baseline_model()
base.summary()
#%%
opt = optimizers.Adam(lr=1e-5)
base.compile(optimizer=opt, loss='mse', metrics=['accuracy'])

base.fit(x_train, y_train, epochs=100, batch_size=128 * 8)

#%%
from sklearn.metrics import r2_score

test = base.predict(x_train, batch_size=1024)

r2 = r2_score(y_train, test)
print('r2={}'.format(r2))

plt.plot(outS.inverse_transform(y_train))
plt.plot(outS.inverse_transform(test), 'rd')
# plt.ylim(-0.3,-0.1)
plt.ylim(-1e-4, 1e-4)
plt.title('r2={}'.format(r2))
plt.show()

#%%
from scipy.integrate import odeint


def dydt(t, x):
    x = np.asarray(x).reshape(1, -1)
    out = outS.inverse_transform(base.predict(inS.transform(x)))

    return out.flatten()


sol2 = solve_ivp(
    dydt,
    (0, 2 / delta),
    [org[0]],
    # t_eval=t,
    method='RK45',
    # method='BDF',
    # max_step=delta,
    rtol=1e-4)

plt.plot(sol2.t, sol2.y[0], 'rd')
plt.title('ODENet steps:{}'.format(sol2.y.shape[1]))
plt.savefig('./fig/flame_ODENet')
plt.show()

#%%
md = 'BDF'
for md in ['RK45', 'BDF']:
    sol = solve_ivp(
        f1,
        (0, 2 / delta),
        [org[0]],
        method=md,
        # max_step=delta,
        rtol=1e-4)

    plt.plot(sol.t, sol.y[0], 'rd')
    plt.title('{} steps:{}'.format(md, sol.y.shape[1]))
    plt.savefig('./fig/flame_{}'.format(md))
    plt.show()

#%%
