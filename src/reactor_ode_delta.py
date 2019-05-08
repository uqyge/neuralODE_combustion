import time
import multiprocessing as mp
import pandas as pd
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

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
        self.gas.set_unnormalized_mass_fractions(y[1:])
        # self.gas.set_unnormalized_mole_fractions(y[1:])
        self.gas.TP = y[0], self.P
        rho = self.gas.density

        wdot = self.gas.net_production_rates
        dTdt = - (np.dot(self.gas.partial_molar_enthalpies, wdot) /
                  (rho * self.gas.cp))
        dYdt = wdot * self.gas.molecular_weights / rho
        # dYdt = wdot /rho

        return np.hstack((dTdt, dYdt))


def ignite_f(ini):
    temp = ini[0]
    n_fuel = ini[1]
    fuel = ini[2]

    train_org = []
    train_new = []

    t_end = 1e-3

    # dt_dict = [5e-7, 7e-7, 1e-6, 1.5e-6]
    # dt_dict = [0.8e-6, 1e-6, 1.2e-6]
    # dt_ini_dict = [0]
    # dt_ini_dict.extend(np.random.random(5))
    num = 10
    dt_ini_dict =[]
    dt_ini_dict.extend([x/num for x in range(num)])
    for ini in dt_ini_dict:
        if fuel == 'H2':
            # gas = ct.Solution('./data/Boivin_newTherm.cti')
            gas = ct.Solution('./data/h2_sandiego.cti')
        if fuel == 'CH4':
            gas = ct.Solution('./data/grimech12.cti')
            # gas = ct.Solution('gri30.xml')
        P = ct.one_atm

        gas.TPX = temp, P, fuel + ':' + str(n_fuel) + ',O2:1,N2:4'
        y0 = np.hstack((gas.T, gas.Y))
        # x0 = np.hstack((gas.T, gas.X))
        ode = ReactorOde(gas)
        solver = scipy.integrate.ode(ode)
        solver.set_integrator('vode', method='bdf', with_jacobian=True)
        solver.set_initial_value(y0, 0.0)
        # solver.set_initial_value(x0, 0.0)
        dt_base = 1e-6
        while solver.successful() and solver.t < t_end:

            if solver.t == 0:
                # dt_ini = np.random.random_sample() * 1e-6
                dt_ini = ini * dt_base
                solver.integrate(solver.t + dt_ini)

            dt = dt_base * (0.8+round(0.4*np.random.random(),2))
            state_org = np.hstack(
                [gas[gas.species_names].concentrations, np.dot(gas.partial_molar_enthalpies, gas[gas.species_names].X),
                 gas.T, gas.density, gas.cp, dt, n_fuel])

            solver.integrate(solver.t + dt)

            gas.TPY = solver.y[0], P, solver.y[1:]
            # gas.TPX = solver.y[0], P, solver.y[1:]

            # Extract the state of the reactor
            state_new = np.hstack(
                [gas[gas.species_names].concentrations, np.dot(gas.partial_molar_enthalpies, gas[gas.species_names].X),
                 gas.T, gas.density, gas.cp, dt, n_fuel])

            # state_new = np.hstack([gas[gas.species_names].Y])
            state_res = state_new - state_org
            res = abs(state_res[state_org != 0] / state_org[state_org != 0])
            # res[res==np.inf]=0
            # res = np.nan_to_num(res)
            # res=res[res!=0]
            # print(res.max())

            # Update the sample
            train_org.append(state_org)
            train_new.append(state_new)

            # if (abs(state_res.max() / state_org.max()) < 1e-5 and (solver.t / dt) > 200):
            thres = ini*1e-2
            if ini == 0:
                thres = 1e-3
            if ((res[:-3].mean() < thres and (solver.t / dt) > 50)) or (gas['H2'].X < 0.005 or gas['H2'].X > 0.995):
                # if res.max() < 1e-5:
                break

    return train_org, train_new


def ignite_post(ini):
    train_org = []
    train_new = []

    temp = ini[0]
    n_fuel = ini[1]
    fuel = ini[2]

    t_end = 1e-3
    dt = 1e-6
    # dt = 5e-7

    if fuel == 'H2':
        # gas = ct.Solution('./data/Boivin_newTherm.cti')
        gas = ct.Solution('./data/h2_sandiego.cti')
    if fuel == 'CH4':
        gas = ct.Solution('./data/grimech12.cti')
        # gas = ct.Solution('gri30.xml')
    # P = ct.one_atm
    #
    # gas.TPX = temp, P, fuel + ':' + str(n_fuel) + ',O2:1,N2:4'
    # y0 = np.hstack((gas.T, gas.Y))
    # # x0 = np.hstack((gas.T, gas.X))
    # ode = ReactorOde(gas)
    # solver = scipy.integrate.ode(ode)
    # solver.set_integrator('vode', method='bdf', with_jacobian=True)
    # solver.set_initial_value(y0, 0.0)
    # solver.set_initial_value(x0, 0.0)

    for step_ini in [1]:
        P = ct.one_atm
        gas.TPX = temp, P, fuel + ':' + str(n_fuel) + ',O2:1,N2:4'
        y0 = np.hstack((gas.T, gas.Y))
        # x0 = np.hstack((gas.T, gas.X))
        ode = ReactorOde(gas)
        solver = scipy.integrate.ode(ode)
        solver.set_integrator('vode', method='bdf', with_jacobian=True)
        solver.set_initial_value(y0, 0.0)
        while solver.successful() and solver.t < t_end:

            if solver.t == 0:
                dt_ini = step_ini * 1e-7
                solver.integrate(solver.t + dt_ini)
                gas.TPY = solver.y[0], P, solver.y[1:]

            state_org = np.hstack(
            [gas[gas.species_names].concentrations, np.dot(gas.partial_molar_enthalpies, gas[gas.species_names].X),
                 gas.T, gas.density, gas.cp, dt])

            solver.integrate(solver.t + dt)
            gas.TPY = solver.y[0], P, solver.y[1:]
            # gas.TPX = solver.y[0], P, solver.y[1:]

            # Extract the state of the reactor
            state_new = np.hstack(
                [gas[gas.species_names].concentrations, np.dot(gas.partial_molar_enthalpies, gas[gas.species_names].X),
                 gas.T, gas.density, gas.cp, dt])

            # state_new = np.hstack([gas[gas.species_names].Y])
            state_res = state_new - state_org
            res = abs(state_res[state_org != 0] / state_org[state_org != 0])
            # res[res==np.inf]=0
            # res = np.nan_to_num(res)
            # res=res[res!=0]
            # print(res.max())

            # Update the sample
            train_org.append(state_org)
            train_new.append(state_new)

            # if (abs(state_res.max() / state_org.max()) < 1e-5 and (solver.t / dt) > 200):
            if ((res[:-3].mean() < 1e-3 and (solver.t / dt) > 50)) or (gas['H2'].X < 0.005 or gas['H2'].X > 0.995):
                # if res.max() < 1e-5:
                print(res.max(), "Y_H2=", gas['H2'].Y)
                break

    return train_org, train_new


def data_gen_f(ini_Tn, fuel):
    if fuel == 'H2':
        # gas = ct.Solution('./data/Boivin_newTherm.cti')
        gas = ct.Solution('./data/h2_sandiego.cti')
    if fuel == 'CH4':
        gas = ct.Solution('./data/grimech12.cti')

    print("multiprocessing:", end='')
    t_start = time.time()
    p = mp.Pool(processes=mp.cpu_count())

    ini = [(x[0], x[1], fuel) for x in ini_Tn]
    # training_data = p.map(ignite_random_x, ini)
    training_data = p.map(ignite_f, ini)
    p.close()

    org, new = zip(*training_data)

    org = np.concatenate(org)
    new = np.concatenate(new)

    columnNames = gas.species_names
    columnNames = columnNames + ['Hs']
    columnNames = columnNames + ['T']
    columnNames = columnNames + ['Rho']
    columnNames = columnNames + ['cp']
    columnNames = columnNames + ['dt']
    columnNames = columnNames + ['f']

    train_org = pd.DataFrame(data=org, columns=columnNames)
    train_new = pd.DataFrame(data=new, columns=columnNames)

    t_end = time.time()
    print(" %8.3f seconds" % (t_end - t_start))

    return train_org, train_new


if __name__ == "__main__":
    ini_T = np.linspace(1201, 1501, 1)
    ini = [(temp, 1) for temp in ini_T]
    # ini = ini + [(temp, 10) for temp in ini_T]
    a, b = data_gen_f(ini, 'H2')
    plt.plot(a['H2'])
    plt.show()
