import time
import multiprocessing as mp
import pandas as pd
import dask.dataframe as dd
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import subprocess
import os

import cantera as ct

print("Running Cantera version: {}".format(ct.__version__))


class ReactorOde(object):
    def __init__(self, gas):
        # Parameters of the ODE system and auxiliary data are stored in the
        # ReactorOde object.
        self.gas = gas
        self.P = gas.P

    def __call__(self, t, y):
        """the ODE function, y' = f(t,y) """

        # State vector is [T, Y_1, Y_2, ... Y_K]
        # self.gas.set_unnormalized_mass_fractions(y[1:])
        self.gas.set_unnormalized_mole_fractions(y[1:])
        self.gas.TP = y[0], self.P
        rho = self.gas.density

        wdot = self.gas.net_production_rates
        dTdt = - (np.dot(self.gas.partial_molar_enthalpies, wdot) /
                  (rho * self.gas.cp))
        # dYdt = wdot * self.gas.molecular_weights / rho
        dYdt = wdot / rho

        return np.hstack((dTdt, dYdt))

def one_step(ini):
    train_org = []
    train_new = []

    temp = ini[0]
    Y_ini = ini[1]
    fuel = ini[2]

    dt = 1e-6

    if fuel == 'H2':
        # gas = ct.Solution('./data/Boivin_newTherm.cti')
        gas = ct.Solution('./data/h2_sandiego.cti')
    if fuel == 'CH4':
        gas = ct.Solution('./data/grimech12.cti')
        # gas = ct.Solution('gri30.xml')
    P = ct.one_atm

    # gas.TPX = temp, P, fuel + ':' + str(n_fuel) + ',O2:1,N2:4'
    gas.TP = temp, P
    gas.Y=Y_ini
    # y0 = np.hstack((gas.T, gas.Y))
    x0 = np.hstack((gas.T, gas.X))
    ode = ReactorOde(gas)
    solver = scipy.integrate.ode(ode)
    solver.set_integrator('vode', method='bdf', with_jacobian=True)
    # solver.set_initial_value(y0, 0.0)
    solver.set_initial_value(x0, 0.0)

    # state_org = np.hstack([gas[gas.species_names].Y, gas.T])
    state_org = np.hstack([gas[gas.species_names].X, gas.T, dt])

    solver.integrate(solver.t + dt)
    # gas.TPY = solver.y[0], P, solver.y[1:]
    gas.TPX = solver.y[0], P, solver.y[1:]

    # Extract the state of the reactor
    state_new = np.hstack([gas[gas.species_names].X, gas.T, dt])

    # state_new = np.hstack([gas[gas.species_names].Y, gas.T])

    # Update the sample
    train_org.append(state_org)
    train_new.append(state_new)

    return train_org, train_new


def one_step_pro(ini):
    train_org = []
    train_new = []

    temp = ini[0]
    Y_ini = ini[1]
    fuel = ini[2]

    dt = 1e-6

    if fuel == 'H2':
        # gas = ct.Solution('./data/Boivin_newTherm.cti')
        gas = ct.Solution('./data/h2_sandiego.cti')
    if fuel == 'CH4':
        gas = ct.Solution('./data/grimech12.cti')
        # gas = ct.Solution('gri30.xml')
    P = ct.one_atm

    # gas.TPX = temp, P, fuel + ':' + str(n_fuel) + ',O2:1,N2:4'
    for temp_org,Y_org in zip(temp,Y_ini):
        gas.TP = temp_org, P
        gas.Y=Y_org

        # y0 = np.hstack((gas.T, gas.Y))
        x0 = np.hstack((gas.T, gas.X))
        ode = ReactorOde(gas)
        solver = scipy.integrate.ode(ode)
        solver.set_integrator('vode', method='bdf', with_jacobian=True)
        # solver.set_initial_value(y0, 0.0)
        solver.set_initial_value(x0, 0.0)

        # state_org = np.hstack([gas[gas.species_names].Y, gas.T])
        state_org = np.hstack([gas[gas.species_names].X, gas.T, dt])

        solver.integrate(solver.t + dt)
        # gas.TPY = solver.y[0], P, solver.y[1:]
        gas.TPX = solver.y[0], P, solver.y[1:]

        # Extract the state of the reactor
        state_new = np.hstack([gas[gas.species_names].X, gas.T, dt])

        # state_new = np.hstack([gas[gas.species_names].Y, gas.T])

        # Update the sample
        train_org.append(state_org)
        train_new.append(state_new)

    return train_org, train_new

def data_gen(ini_Tn, fuel):
    gas = ct.Solution('./data/h2_sandiego.cti')

    print("multiprocessing:", end='')
    t_start = time.time()
    p = mp.Pool(processes=mp.cpu_count())


    ini = [(x[0], x[1], fuel) for x in ini_Tn]
    # training_data = p.map(ignite_random_x, ini)
    training_data = p.map(one_step_pro, ini)
    p.close()

    org, new = zip(*training_data)

    org = np.concatenate(org)
    new = np.concatenate(new)

    columnNames = gas.species_names
    columnNames = columnNames + ['T']
    columnNames = columnNames + ['dt']

    train_org = pd.DataFrame(data=org, columns=columnNames)
    train_new = pd.DataFrame(data=new, columns=columnNames)

    t_end = time.time()
    print(" %8.3f seconds" % (t_end - t_start))

    return train_org, train_new


def fm_data_gen():
    gas = ct.Solution('./data/h2_sandiego.cti')

    t_start = time.time()
    # df = pd.DataFrame()
    # for file in os.scandir('./data/fm'):
    #     # subprocess.call(['sed','-i', 's/[ []/[/g', file.path])
    #     # subprocess.call(['sed','-i', 's/Y-//g', file.path])
    #
    #     df_tmp = pd.read_csv(file.path, delimiter=r'\s+', skiprows=1)
    #     df=df.append(df_tmp)

    # read in parallel
    df = dd.read_csv('./data/fm/*.kg',delimiter=r'\s+',skiprows=1)
    df = df.compute()
    t_end = time.time()
    print(" %8.3f seconds" % (t_end - t_start))

    df = df[df['Z'] > 0.005]
    df = df[df['Z'] < 0.995]
    df=df[df['temperature[[K]']>900]
    Y_sp = df[gas.species_names]
    T = df['temperature[[K]']

    #ini=[ (a,b) for a,b in zip(T,Y_sp.values)]
    ini = [(a, b) for a, b in zip(np.array_split(T,mp.cpu_count()),
                                  np.array_split(Y_sp.values,mp.cpu_count()))]

    train_org, train_new = data_gen(ini, 'H2')
    return train_org, train_new

if __name__ == '__main__':
    gas = ct.Solution('./data/h2_sandiego.cti')

    P = ct.one_atm
    t_start = time.time()
    # df = pd.DataFrame()
    # for file in os.scandir('./data/fm'):
    #     # subprocess.call(['sed','-i', 's/[ []/[/g', file.path])
    #     # subprocess.call(['sed','-i', 's/Y-//g', file.path])
    #
    #     df_tmp = pd.read_csv(file.path, delimiter=r'\s+', skiprows=1)
    #     df=df.append(df_tmp)

    # read in parallel
    # for file in os.scandir('./data/fm'):
    #     subprocess.call(['sed','-i', 's/[ []/[/g', file.path])
    #     subprocess.call(['sed','-i', 's/Y-//g', file.path])

    df = dd.read_csv('./data/fm/*.kg',delimiter=r'\s+',skiprows=1)
    df = df.compute()
    t_end = time.time()
    print(" %8.3f seconds" % (t_end - t_start))

    df = df[df['Z'] > 0.005]
    df = df[df['Z'] < 0.995]
    df=df[df['temperature[[K]']>900]
    Y_sp = df[gas.species_names]
    T = df['temperature[[K]']

    ini = [(a, b) for a, b in zip(T, Y_sp.values)]
    ini_pro = [(a, b) for a, b in zip(np.array_split(T,4), np.array_split(Y_sp.values,4))]
    a, b = data_gen(ini_pro, 'H2')