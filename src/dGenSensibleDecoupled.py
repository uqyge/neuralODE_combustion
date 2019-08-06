import pandas as pd
import dask
import dask.dataframe as dd
from dask.delayed import delayed

import time
import cantera as ct
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt


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
        # self.gas.Y = y[1:]
        self.gas.TP = y[0], self.P
        rho = self.gas.density

        wdot = self.gas.net_production_rates
        dTdt = -(np.dot(self.gas.partial_molar_enthalpies, wdot) /
                 (rho * self.gas.cp))
        dYdt = wdot * self.gas.molecular_weights / rho

        return np.hstack((dTdt, dYdt))


def ignite_f_decoupled(ini):
    temp, n_fuel, fuel = ini

    train_c = []
    train_wdot = []
    phi = []

    T_ref = 298.15
    t_end = 1e-3

    dt_dict = [1e-7]
    for dt in dt_dict:
        if fuel == 'H2':
            try:
                gas = ct.Solution('../data/connaire.cti')
                gas_h0 = ct.Solution('../data/connaire.cti')
            except:
                gas = ct.Solution('./data/connaire.cti')
                gas_h0 = ct.Solution('./data/connaire.cti')

        if fuel == 'CH4':
            gas = ct.Solution('../data/grimech12.cti')
            gas_h0 = ct.Solution('../data/grimech12.cti')

        P = ct.one_atm
        gas.TPX = temp, P, fuel + ':' + str(n_fuel) + ',O2:1,N2:3.76'

        # print(gas.get_equivalence_ratio())
        phi.append(gas.get_equivalence_ratio())
        Ha = np.dot(gas.partial_molar_enthalpies,
                    gas.Y / gas.molecular_weights)

        gas_h0.TPY = T_ref, P, gas.Y
        hc = (gas_h0.partial_molar_enthalpies / gas_h0.molecular_weights)

        y0 = np.hstack((gas.T, gas.Y))
        ode = ReactorOde(gas)
        solver = scipy.integrate.ode(ode)
        solver.set_integrator('vode',
                              atol=1e-12,
                              rtol=1e-10,
                              method='adams',
                              with_jacobian=True)
        solver.set_initial_value(y0, 0.0)

        v = 120
        dt_base = dt
        while solver.successful() and solver.t < t_end:
            if solver.t == 0:
                # dt_ini = np.random.random_sample() * dt_base
                dt_ini = dt_base
                solver.integrate(solver.t + dt_ini)
                gas.TPY = solver.y[0], P, solver.y[1:]
                # gas.TP = solver.y[0], P
                # gas.set_unnormalized_mass_fractions(solver.y[1:])

            # # sensible enthalpy
            # gas_h0.TPY = T_ref, P, gas.Y
            # H0 = np.dot(gas_h0.partial_molar_enthalpies,
            #             (gas_h0.Y / gas_h0.molecular_weights))
            # Ha = np.dot(gas.partial_molar_enthalpies,
            #             gas.Y / gas.molecular_weights)
            # hs_org = (Ha - H0)

            # # convection
            # convY = np.maximum(gas.Y - gas.Y * v * dt, 1e-20)
            # convT = gas.T - hs_org * v * dt / gas.cp
            # convP = gas.P - gas.P * v * dt
            # gas.TP = convT, convP
            # # gas.set_unnormalized_mass_fractions(convY)
            # if(v>0){

            # }

            gas_h0.TPY = T_ref, P, gas.Y
            H0 = np.dot(gas_h0.partial_molar_enthalpies,
                        (gas_h0.Y / gas_h0.molecular_weights))
            # Ha = np.dot(gas.partial_molar_enthalpies,
            #             gas.Y / gas.molecular_weights)
            hs_org = (Ha - H0)
            hs_density = hs_org * gas.density

            state_org = np.hstack([
                gas[gas.species_names].concentrations, hs_density, gas.T,
                gas.density, gas.cp, dt, n_fuel
            ])
            w_dot = gas[gas.species_names].net_production_rates
            # hs_dot = np.dot(gas.partial_molar_enthalpies, -w_dot) / gas.density
            # BUG
            hs_dot = np.dot(gas_h0.partial_molar_enthalpies, -w_dot)
            T_dot = hs_dot / (gas.density * gas.cp)
            # T_dot = hs_dot / gas.cp

            T_org = gas.T
            w_tracker = max(abs(w_dot[:-1] / gas.concentrations[:-1]))
            # print(max(w_tracker[:-1]))

            state_c = np.hstack([
                gas[gas.species_names].concentrations, hs_org, gas.T,
                gas.density, gas.cp, dt, n_fuel
            ])
            train_c.append(state_c)

            # dt = dt_base * (0.9 + round(0.2 * np.random.random(), 2))
            dt = dt_base
            # print("base:", dt)
            if (w_tracker * dt) > 0.2:
                dt = dt / 10
            # print(w_tracker * dt_base)
            # print("a:", dt)

            solver.integrate(solver.t + dt)
            gas.TPY = solver.y[0], P, solver.y[1:]
            gas_h0.TPY = T_ref, P, gas.Y

            # gas.TP = solver.y[0], P
            # gas.set_unnormalized_mass_fractions(solver.y[1:])
            # gas_h0.TP = T_ref, P
            # gas_h0.set_unnormalized_mass_fractions(solver.y[1:])

            H0 = np.dot(gas_h0.partial_molar_enthalpies,
                        (gas_h0.Y / gas_h0.molecular_weights))
            hs_new = (Ha - H0)
            hs_density = hs_new * gas.density

            # Extract the state of the reactor
            state_new = np.hstack([
                gas[gas.species_names].concentrations, hs_density, gas.T,
                gas.density, gas.cp, dt, n_fuel
            ])

            state_res = state_new - state_org
            res = abs(
                state_res[state_org != 0] / state_org[state_org != 0]) / dt

            state_wdot = np.hstack([w_dot, hs_dot, T_dot])
            train_wdot.append(state_wdot)

            # add convection after reaction
            gas.Y = gas.Y - T_dot * 1e-8 * v * dt * gas.Y
            # gas.TP = gas.T - T_dot * 1e-8 * v * dt * gas.T, P

            if ((res.max() < 1e2 and (solver.t / dt) > 100)):
                break

    return train_c, train_wdot, phi


def dataGeneration():
    dask.config.set(scheduler='processes')

    # T = np.random.rand(2) * 1200 + 1001
    T = np.linspace(1201, 2201, 1)

    # n_s = np.random.rand(10) * 30 + 0.1
    # n_l = np.random.rand(30) * 30
    n_s = np.linspace(2, 8, 1)
    n_l = np.linspace(0.3, 30, 1)

    n = np.unique(np.concatenate((n_s, n_l)))[1:]
    # n = n[n > 0.4]
    n = n[n >= 0.1]

    XX, YY = np.meshgrid(T, n)
    ini = np.concatenate((XX.reshape(-1, 1), YY.reshape(-1, 1)), axis=1)
    # print(ini)

    s = time.time()

    a = [delayed(ignite_f_decoupled)([x[0], x[1], 'H2']) for x in ini]
    a = dask.compute(*a)

    print('a[0][0]={}'.format(len(a[0][0])))
    e = time.time()
    print('Creating {} sets, taking {}s'.format(len(a), e - s))

    org = np.concatenate([x[0] for x in a])
    wdot = np.concatenate([x[1] for x in a])
    phi = np.concatenate([x[2] for x in a])

    try:
        gas = ct.Solution('./data/connaire.cti')
    except:
        gas = ct.Solution('../data/connaire.cti')

    columnNames = gas.species_names
    columnNames = columnNames + ['Hs']
    columnNames = columnNames + ['Temp']
    wdotNames = columnNames
    columnNames = columnNames + ['rho']
    columnNames = columnNames + ['cp']
    columnNames = columnNames + ['dt']
    columnNames = columnNames + ['f']

    train_org = pd.DataFrame(data=org, columns=columnNames)
    train_wdot = pd.DataFrame(data=wdot, columns=wdotNames)

    s = time.time()

    train_org.to_hdf('tmp.h5', key='c', format='table')
    train_wdot.to_hdf('tmp.h5', key='wdot', format='table')
    sp = "H2O2"
    print("{} wdot min:{}".format(sp, train_wdot[sp].min()))
    e = time.time()
    print('saving {} takes {}s'.format(train_wdot.shape, (e - s)))
    # plt.plot(phi)
    # return train_org
    return train_org, train_wdot


def ignite_post(ini):
    temp = ini[0]
    n_fuel = ini[1]
    fuel = ini[2]

    train_c = []
    train_wdot = []

    t_end = 1e-3
    T_ref = 298.15

    dt_dict = [ini[3]]
    for dt in dt_dict:
        if fuel == 'H2':
            # gas = ct.Solution('./data/Boivin_newTherm.cti')
            # gas = ct.Solution('./data/h2_sandiego.cti')
            try:
                gas = ct.Solution('../data/connaire.cti')
                gas_h0 = ct.Solution('../data/connaire.cti')
            except:
                gas = ct.Solution('./data/connaire.cti')
                gas_h0 = ct.Solution('./data/connaire.cti')

        if fuel == 'CH4':
            gas = ct.Solution('./data/grimech12.cti')
            gas_h0 = ct.Solution('./data/grimech12.cti')
            # gas = ct.Solution('gri30.xml')
        P = ct.one_atm

        gas.TPX = temp, P, fuel + ':' + str(n_fuel) + ',O2:1,N2:3.728'
        Ha = np.dot(gas.partial_molar_enthalpies,
                    gas.Y / gas.molecular_weights)

        y0 = np.hstack((gas.T, gas.Y))
        ode = ReactorOde(gas)
        solver = scipy.integrate.ode(ode)
        solver.set_integrator('vode', method='bdf', with_jacobian=True)
        solver.set_initial_value(y0, 0.0)

        while solver.successful() and solver.t < t_end:
            if solver.t == 0:
                dt_ini = dt
                solver.integrate(solver.t + dt_ini)
                gas.TPY = solver.y[0], P, solver.y[1:]

            # Ha = np.dot(gas.partial_molar_enthalpies,
            #             gas.Y / gas.molecular_weights)
            # print('t:{}'.format(solver.t))
            gas_h0.TPY = T_ref, P, gas.Y
            H0 = np.dot(gas_h0.partial_molar_enthalpies,
                        (gas_h0.Y / gas_h0.molecular_weights))
            # hs = (Ha - H0) * gas.density
            hs_org = (Ha - H0)
            hs_density = hs_org * gas.density
            # print("enthalpy of formation @{}:   {}".format(gas_h0.T, h0))
            # print("absolute enthalpy @{}:   {}".format(gas.T, ha))
            # print("sensible enthalpy org @{}:   {}".format(gas.T, hs))

            state_org = np.hstack([
                gas[gas.species_names].concentrations, hs_density, gas.T,
                gas.density, gas.cp, dt, n_fuel
            ])
            w_dot = gas[gas.species_names].net_production_rates
            T_org = gas.T
            hs_dot = np.dot(gas.partial_molar_enthalpies, -w_dot) / gas.density
            state_c = np.hstack([
                gas[gas.species_names].concentrations, hs_org, gas.T,
                gas.density, gas.cp, dt, n_fuel
            ])
            train_c.append(state_c)

            solver.integrate(solver.t + dt)
            gas.TPY = solver.y[0], P, solver.y[1:]
            # Ha = np.dot(gas.partial_molar_enthalpies,
            #             gas.Y / gas.molecular_weights)

            gas_h0.TPY = T_ref, P, gas.Y
            H0 = np.dot(gas_h0.partial_molar_enthalpies,
                        (gas_h0.Y / gas_h0.molecular_weights))
            hs_new = (Ha - H0)
            hs_density = hs_new * gas.density

            # print("sensible enthalpy new @{}:   {}".format(gas.T, hs))
            # Extract the state of the reactor
            state_new = np.hstack([
                gas[gas.species_names].concentrations, hs_density, gas.T,
                gas.density, gas.cp, dt, n_fuel
            ])

            # state_new = np.hstack([gas[gas.species_names].Y])
            state_res = state_new - state_org
            res = abs(state_res[:-2][state_org[:-2] != 0] /
                      state_org[:-2][state_org[:-2] != 0]) / dt

            # Update the sample

            # hs_dot = (hs_new - hs_org) / dt
            state_wdot = np.hstack([w_dot, hs_dot, (gas.T - T_org) / dt])
            train_wdot.append(state_wdot)

            # if (abs(state_res.max() / state_org.max()) < 1e-5 and (solver.t / dt) > 200):
            if ((res.max() < 1e3 and
                 (solver.t / dt) > 50)) or (gas['H2'].X < 0.005
                                            or gas['H2'].X > 0.995):
                # if res.max() < 1e-5:
                # # print(res)
                # print(state_org)
                break

    return train_c, train_wdot


def ignite_step(ini, gas, gas_h0):
    temp, Y_ini, fuel, dt = ini
    n_fuel = -1

    T_ref = 298.15
    t_end = dt

    train_c = []
    train_wdot = []

    P = ct.one_atm

    gas.TPY = temp, P, Y_ini
    Ha = np.dot(gas.partial_molar_enthalpies, gas.Y / gas.molecular_weights)

    y0 = np.hstack((gas.T, gas.Y))
    ode = ReactorOde(gas)
    solver = scipy.integrate.ode(ode)
    solver.set_integrator('vode', method='bdf', with_jacobian=True)
    solver.set_initial_value(y0, 0.0)

    while solver.successful() and solver.t < t_end:

        gas_h0.TPY = T_ref, P, gas.Y
        H0 = np.dot(gas_h0.partial_molar_enthalpies,
                    (gas_h0.Y / gas_h0.molecular_weights))
        hs_org = (Ha - H0)
        hs_density = hs_org * gas.density

        w_dot = gas[gas.species_names].net_production_rates
        hs_dot = np.dot(gas_h0.partial_molar_enthalpies, -w_dot)
        T_dot = hs_dot / (gas.density * gas.cp)

        state_c = np.hstack(
            [gas[gas.species_names].concentrations, hs_org, gas.T])
        train_c.append(state_c)

        solver.integrate(solver.t + dt)
        gas.TPY = solver.y[0], P, solver.y[1:]
        gas_h0.TPY = T_ref, P, gas.Y
        H0 = np.dot(gas_h0.partial_molar_enthalpies,
                    (gas_h0.Y / gas_h0.molecular_weights))

        state_wdot = np.hstack([w_dot, hs_dot, T_dot])
        train_wdot.append(state_wdot)

    return train_c, train_wdot


def test_data(temp, n_fuel, columns, dt):
    ode_o, ode_n = ignite_post((temp, n_fuel, 'H2', dt))
    ode_o = np.asarray(ode_o)
    ode_n = np.asarray(ode_n)
    #     ode_o = np.append(ode_o, n_fuel * np.ones((ode_o.shape[0], 1)), axis=1)
    #     ode_n = np.append(ode_n, n_fuel * np.ones((ode_n.shape[0], 1)), axis=1)
    ode_o = pd.DataFrame(data=ode_o, columns=columns)
    ode_n = pd.DataFrame(data=ode_n, columns=columns)

    # idx_test = (ode_o > 0).all(1)
    # ode_o = ode_o[idx_test]
    # ode_n = ode_n[idx_test]

    return ode_o, ode_n


if __name__ == '__main__':
    t_org, t_wdot = dataGeneration()
    # plt.plot(t_org['T'])
    plt.plot(t_wdot['H2O2'])
    # c, w = ignite_post([1401, 2, 'H2', 1e-6])

    # try:
    #     gas = ct.Solution('./data/connaire.cti')
    # except:
    #     gas = ct.Solution('../data/connaire.cti')

    # # gas = ct.Solution('../data/grimech12.cti')
    # columnNames = gas.species_names
    # columnNames = columnNames + ['Hs']
    # columnNames = columnNames + ['T']
    # wdotNames = columnNames
    # columnNames = columnNames + ['Rho']
    # columnNames = columnNames + ['cp']
    # columnNames = columnNames + ['dt']
    # columnNames = columnNames + ['f']
    # c = pd.DataFrame(c, columns=columnNames)
    # w = pd.DataFrame(w, columns=wdotNames)
    # plt.plot(w['Hs'])