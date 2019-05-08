import pandas as pd
import dask
from dask.delayed import delayed

import time
import cantera as ct
import numpy as np
import scipy.integrate


class ReactorOde:
    def __init__(self, gas):
        # Parameters of the ODE system and auxiliary data are stored in the
        # ReactorOde object.
        self.gas = gas
        self.P = gas.P

    def __call__(self, t, y):
        """the ODE function, y' = f(t,y) """

        # State vector is [T, Y_1, Y_2, ... Y_K]
        self.gas.set_unnormalized_mass_fractions(y[1:])
        self.gas.TP = y[0], self.P
        rho = self.gas.density

        wdot = self.gas.net_production_rates
        dTdt = - (np.dot(self.gas.partial_molar_enthalpies, wdot) /
                  (rho * self.gas.cp))
        dYdt = wdot * self.gas.molecular_weights / rho

        return np.hstack((dTdt, dYdt))


def ignite_f(ini):
    temp = ini[0]
    n_fuel = ini[1]
    fuel = ini[2]

    train_org = []
    train_new = []

    t_end = 1e-3

    dt_dict = [1e-7]
    for dt in dt_dict:
        if fuel == 'H2':
            # gas = ct.Solution('./data/Boivin_newTherm.cti')
            gas = ct.Solution('../data/h2_sandiego.cti')
        if fuel == 'CH4':
            gas = ct.Solution('../data/grimech12.cti')
            # gas = ct.Solution('gri30.xml')
        P = ct.one_atm

        gas.TPX = temp, P, fuel + ':' + str(n_fuel) + ',O2:1,N2:4'
        y0 = np.hstack((gas.T, gas.Y))
        ode = ReactorOde(gas)
        solver = scipy.integrate.ode(ode)
        solver.set_integrator('vode', method='bdf', with_jacobian=True)
        solver.set_initial_value(y0, 0.0)
        dt_base = dt
        while solver.successful() and solver.t < t_end:

            if solver.t == 0:
                dt_ini = np.random.random_sample() * dt_base
                solver.integrate(solver.t + dt_ini)

            dt = dt_base * (0.9+round(0.2*np.random.random(), 2))
            state_org = np.hstack(
                [gas[gas.species_names].concentrations, np.dot(gas.partial_molar_enthalpies, gas[gas.species_names].X),
                 gas.T, gas.density, gas.cp, dt, n_fuel])

            solver.integrate(solver.t + dt)

            gas.TPY = solver.y[0], P, solver.y[1:]

            # Extract the state of the reactor
            state_new = np.hstack(
                [gas[gas.species_names].concentrations, np.dot(gas.partial_molar_enthalpies, gas[gas.species_names].X),
                 gas.T, gas.density, gas.cp, dt, n_fuel])

            # state_new = np.hstack([gas[gas.species_names].Y])
            state_res = state_new - state_org
            res = abs(state_res[state_org != 0] / state_org[state_org != 0])/dt
            # res[res==np.inf]=0
            # res = np.nan_to_num(res)
            # res=res[res!=0]
            # print(res.max())

            # Update the sample
            train_org.append(state_org)
            train_new.append(state_new)

            # if (abs(state_res.max() / state_org.max()) < 1e-5 and (solver.t / dt) > 200):
            if ((res.max() < 1e-3 and (solver.t / dt) > 50)) or (gas['H2'].X < 0.005 or gas['H2'].X > 0.995):
                # if res.max() < 1e-5:
                break

    return train_org, train_new


def dataGeneration():
    T = np.random.rand(2) * 1200 + 1001

    # n_s = np.random.rand(30) * 30 + 0.1
    # n_l = np.random.rand(30) * 30
    n_s = np.linspace(0, 8, 20)
    n_l = np.linspace(0, 30, 30)

    n = np.unique(np.concatenate((n_s, n_l)))[1:]
    n = n[n > 0.4]

    XX, YY = np.meshgrid(T, n)
    ini = np.concatenate((XX.reshape(-1, 1), YY.reshape(-1, 1)), axis=1)
    # print(ini)

    dask.config.set(scheduler='processes')

    s = time.time()

    a = [delayed(ignite_f)([x[0], x[1], 'H2']) for x in ini]
    a = dask.compute(*a)

    print('a[0][0]={}'.format(len(a[0][0])))
    e = time.time()
    print('There are {} sets.'.format(len(a)), e-s)

    org = np.concatenate([x[0] for x in a])
    new = np.concatenate([x[1] for x in a])

    gas = ct.Solution('../data/h2_sandiego.cti')
    columnNames = gas.species_names
    columnNames = columnNames + ['Hs']
    columnNames = columnNames + ['T']
    columnNames = columnNames + ['Rho']
    columnNames = columnNames + ['cp']
    columnNames = columnNames + ['dt']
    columnNames = columnNames + ['f']

    train_org = pd.DataFrame(data=org, columns=columnNames)
    train_new = pd.DataFrame(data=new, columns=columnNames)

    df = pd.concat([train_org, train_new])
    key = 'all_the_things'
    df.to_hdf('merged.h5', key, complib='zlib', complevel=9)


def ignite_post(ini):
    temp = ini[0]
    n_fuel = ini[1]
    fuel = ini[2]

    train_org = []
    train_new = []

    t_end = 1e-3

    dt_dict = [ini[3]]
    for dt in dt_dict:
        if fuel == 'H2':
            # gas = ct.Solution('./data/Boivin_newTherm.cti')
            gas = ct.Solution('./data/h2_sandiego.cti')
        if fuel == 'CH4':
            gas = ct.Solution('./data/grimech12.cti')
            # gas = ct.Solution('gri30.xml')
        P = ct.one_atm

        gas.TPX = temp, P, fuel + ':' + str(n_fuel) + ',O2:1,N2:4'
        y0 = np.hstack((gas.T, gas.Y))
        ode = ReactorOde(gas)
        solver = scipy.integrate.ode(ode)
        solver.set_integrator('vode', method='bdf', with_jacobian=True)
        solver.set_initial_value(y0, 0.0)
#         dt_base = dt
        while solver.successful() and solver.t < t_end:

            if solver.t == 0:
                #                 dt_ini = np.random.random_sample() * 1e-6
                dt_ini = dt
                solver.integrate(solver.t + dt_ini)

#             dt = dt_base * (0.9+round(0.2*np.random.random(),2))
            state_org = np.hstack(
                [gas[gas.species_names].concentrations, np.dot(gas.partial_molar_enthalpies, gas[gas.species_names].X),
                 gas.T, gas.density, gas.cp, dt, n_fuel])

            solver.integrate(solver.t + dt)

            gas.TPY = solver.y[0], P, solver.y[1:]

            # Extract the state of the reactor
            state_new = np.hstack(
                [gas[gas.species_names].concentrations, np.dot(gas.partial_molar_enthalpies, gas[gas.species_names].X),
                 gas.T, gas.density, gas.cp, dt, n_fuel])

            # state_new = np.hstack([gas[gas.species_names].Y])
            state_res = state_new - state_org
            res = abs(state_res[:-2][state_org[:-2] != 0] /
                      state_org[:-2][state_org[:-2] != 0])/dt

            # Update the sample
            train_org.append(state_org)
            train_new.append(state_new)

            # if (abs(state_res.max() / state_org.max()) < 1e-5 and (solver.t / dt) > 200):
            if ((res.max() < 1e-3 and (solver.t / dt) > 50)) or (gas['H2'].X < 0.005 or gas['H2'].X > 0.995):
                # if res.max() < 1e-5:
                break

    return train_org, train_new


def test_data(temp, n_fuel, columns, dt):
    ode_o, ode_n = ignite_post((temp, n_fuel, 'H2', dt))
    ode_o = np.asarray(ode_o)
    ode_n = np.asarray(ode_n)
#     ode_o = np.append(ode_o, n_fuel * np.ones((ode_o.shape[0], 1)), axis=1)
#     ode_n = np.append(ode_n, n_fuel * np.ones((ode_n.shape[0], 1)), axis=1)
    ode_o = pd.DataFrame(data=ode_o,
                         columns=columns)
    ode_n = pd.DataFrame(data=ode_n,
                         columns=columns)

    idx_test = (ode_o > 1e-9).all(1)
    ode_o = ode_o[idx_test]
    ode_n = ode_n[idx_test]

    return ode_o, ode_n


if __name__ == '__main__':
    dataGeneration()
