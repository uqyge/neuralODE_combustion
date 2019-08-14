import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection, metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler

os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ['KERAS_BACKEND'] = 'cntk'

from keras import backend as K

K.set_floatx('float32')
print("precision: " + K.floatx())
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.models import Model
from keras.layers import Dense, Input, Activation, Average
from keras.callbacks import ModelCheckpoint
from keras import optimizers

from src.res_block import res_block
from src.reactor_ode_delta import data_gen_f
from src.dataScaling import dataScaling
import cantera as ct

print("Running Cantera version: {}".format(ct.__version__))


class classScaler(object):
    def __init__(self):
        self.norm = None
        self.std = None

    def fit_transform(self, input_data):
        self.norm = MinMaxScaler()
        self.std = StandardScaler()
        out = self.std.fit_transform(input_data)
        out = self.norm.fit_transform(out)
        return out

    def transform(self, input_data):
        out = self.std.transform(input_data)
        out = self.norm.transform(out)

        return out


class cluster(object):
    def __init__(self, data, T):
        self.T_ = T
        self.labels_ = np.asarray((data['T'] > self.T_).astype(int))

    def predict(self, input):
        out = (input[:, -1] > self.T_).astype(int)
        return out


class combustionML(object):

    def __init__(self, df_x_input, df_y_target, scaling_case):
        x_train, x_test, y_train, y_test = model_selection.train_test_split(df_x_input, df_y_target,
                                                                            test_size=0.1,
                                                                            random_state=42)

        self.x_scaling = dataScaling()
        self.y_scaling = dataScaling()
        self.x_train = self.x_scaling.fit_transform(x_train, scaling_case['x'])
        self.y_train = self.y_scaling.fit_transform(y_train, scaling_case['y'])
        x_test = self.x_scaling.transform(x_test)

        self.scaling_case = scaling_case
        self.df_x_input = df_x_input
        self.df_y_target = df_y_target
        self.x_test = pd.DataFrame(data=x_test, columns=df_x_input.columns)
        self.y_test = pd.DataFrame(data=y_test, columns=df_y_target.columns)

        self.floatx = 'float32'
        self.dim_input = self.x_train.shape[1]
        self.dim_label = self.y_train.shape[1]

        self.inputs = Input(shape=(self.dim_input,), dtype=self.floatx)

        self.model = None
        self.ensemble_num = 4
        self.model_ensemble = None
        self.history = None
        self.callbacks_list = None
        self.vsplit = None
        self.predict = None

    def res_reg_model(self, model_input, id, n_neurons=200, blocks=2, drop1=0.1, batch_norm=False):
        print('set up ANN :', model_input.dtype)
        x = Dense(n_neurons, name='1_base' + id)(model_input)
        # x = BatchNormalization(axis=-1, name='1_base_bn')(x)
        x = Activation('relu')(x)

        for b in range(blocks):
            x = res_block(x, n_neurons, stage=1, block=str(b) + id, d1=drop1, bn=batch_norm)

        # # second block
        # b2_neurons = int(n_neurons/8)
        # x = Dense(b2_neurons, name='2_base' + id)(x)
        # # x = BatchNormalization(axis=-1, name='1_base_bn')(x)
        # x = Activation('relu')(x)
        #
        # for b in range(blocks):
        #     x = res_block(x, b2_neurons, stage=2, block=str(b) + id, d1=drop1, bn=batch_norm)

        predictions = Dense(self.dim_label, activation='linear')(x)

        model = Model(inputs=model_input, outputs=predictions)
        return model

    def fitModel(self, batch_size=1024, epochs=200, vsplit=0.1, sfl=True, ensemble_num=4):
        self.vsplit = vsplit

        filepath = "./tmp/history/weights.improvement_{val_loss:.4f}_.hdf5"
        checkpoint = ModelCheckpoint(filepath,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='min',
                                     period=5)

        self.callbacks_list = [checkpoint]
        self.history = self.model.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=vsplit,
            verbose=2,
            callbacks=self.callbacks_list,
            shuffle=sfl)

        self.model.save_weights("./tmp/weights.last.hdf5")

        names = []
        for fl in os.listdir('./tmp/history'):
            name = fl.split('_')
            names.append(float(name[1]))
        names.sort()
        self.ensemble_num = min(ensemble_num, len(names))
        for i in range(self.ensemble_num):
            a = name[0] + '_' + format(names[i], '.4f') + '_' + name[2]
            print(a)
            os.rename('./tmp/history/' + a, './tmp/weights_' + str(i) + '.hdf5')

    def prediction(self):
        self.model.load_weights("./tmp/weights_0.hdf5")

        predict = self.model.predict(self.x_test.values)
        predict = self.y_scaling.inverse_transform(predict)
        self.predict = pd.DataFrame(data=predict, columns=self.df_y_target.columns)

        R2_score = abs(metrics.r2_score(predict, self.y_test))
        print(R2_score)
        return R2_score

    def ensemble(self, n_neurons=200, blocks=2, drop1=.0):
        model_last = self.res_reg_model(self.inputs, '_last_', n_neurons=n_neurons, blocks=blocks, drop1=drop1)
        model_last.load_weights("./tmp/weights.last.hdf5")

        models = []
        models.append(model_last)

        for i in range(self.ensemble_num):
            model = self.res_reg_model(self.inputs, str(i), n_neurons=n_neurons, blocks=blocks, drop1=drop1)
            model.load_weights("./tmp/weights_" + str(i) + ".hdf5")
            models.append(model)

        outputs = [model.outputs[0] for model in models]
        y = Average()(outputs)

        self.model_ensemble = Model(inputs=self.inputs, outputs=y, name='ensemble')
        self.model.load_weights("./tmp/weights_0.hdf5")

    def inference(self, x):
        tmp = self.x_scaling.transform(x)
        predict = self.model.predict(tmp, batch_size=1024 * 8 * 2)
        # inverse for out put
        out = self.y_scaling.inverse_transform(predict)
        # eliminate negative values
        out[out < 0] = 0

        return out

    def inference_ensemble(self, x, batch_size=1204):
        tmp = self.x_scaling.transform(x)
        predict = self.model_ensemble.predict(tmp, batch_size=batch_size)
        # inverse for out put
        out = self.y_scaling.inverse_transform(predict)
        # eliminate negative values
        out[out < 0] = 0

        return out

    def plt_acc(self, sp):

        plt.figure()
        plt.plot(self.y_test[sp], self.predict[sp], 'kd', ms=1)
        # plt.axis('tight')
        # plt.axis('equal')

        # plt.axis([train_new[sp].min(), train_new[sp].max(), train_new[sp].min(), train_new[sp].max()], 'tight')
        r2 = round(metrics.r2_score(self.y_test[sp], self.predict[sp]), 6)
        plt.title(sp + ' : r2 = ' + str(r2))
        plt.show()

        t_n = self.y_scaling.transform(self.y_test)
        p_n = self.y_scaling.transform(self.predict)
        t_n = pd.DataFrame(data=t_n, columns=self.df_y_target.columns)
        p_n = pd.DataFrame(data=p_n, columns=self.df_y_target.columns)

        plt.figure()
        plt.plot(t_n[sp], p_n[sp], 'kd', ms=1)
        # plt.axis('tight')
        # plt.axis('equal')

        # plt.axis([train_new[sp].min(), train_new[sp].max(), train_new[sp].min(), train_new[sp].max()], 'tight')
        r2_n = round(metrics.r2_score(t_n[sp], p_n[sp]), 6)
        plt.title(sp + ' nn: r2 = ' + str(r2_n))
        plt.show()

    def plt_loss(self):
        plt.semilogy(self.history.history['loss'])
        if self.vsplit:
            plt.semilogy(self.history.history['val_loss'])
        plt.title('mae')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.show()

    def run(self, hyper):
        print(hyper)
        sgd = optimizers.SGD(lr=0.3, decay=1e-3, momentum=0.9, nesterov=True)
        rms = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
                               epsilon=1e-8, decay=0.0, amsgrad=True)

        self.model = self.res_reg_model(self.inputs, '_base_', n_neurons=hyper[0],
                                        blocks=hyper[1], drop1=hyper[2], batch_norm=False)

        self.model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])

        self.fitModel(epochs=hyper[3], batch_size=1024 * 8 * 4, vsplit=0.2,
                      sfl=False, ensemble_num=self.ensemble_num)

        self.ensemble(n_neurons=hyper[0], blocks=hyper[1], drop1=hyper[2])

        r2 = self.prediction()
        return r2


if __name__ == "__main__":
    T = np.random.rand(20) * 1000 + 1001
    n_s = np.random.rand(20) * 7.6 + 0.4
    n_l = np.random.rand(20) * 40
    # n = np.random.randint(10000, size=2000)
    n = np.concatenate((n_s, n_l))
    XX, YY = np.meshgrid(T, n)
    ini = np.concatenate((XX.reshape(-1, 1), YY.reshape(-1, 1)), axis=1)

    # generate data
    df_x_input_org, df_y_target_org = data_gen_f(ini, 'H2')
    # df_x_input, df_y_target = fm_data_gen()
    # fm_x, fm_y = fm_data_gen()
    # df_x_input.append(fm_x)
    # df_y_target.append(fm_y)
    x_columns = df_x_input_org.columns

    import cantera as ct

    gas = ct.Solution('./data/h2_sandiego.cti')
    P = ct.one_atm
    XT = df_x_input_org.values[:, :-1]
    phi_dot = []
    for i in range(0, XT.shape[0]):
        # for i in range(0, 5):
        gas.TP = XT[i, -1], P
        gas.set_unnormalized_mole_fractions(XT[i, :-1])
        rho = gas.density

        wdot = gas.net_production_rates
        dTdt = - (np.dot(gas.partial_molar_enthalpies, wdot) /
                  (rho * gas.cp))
        phi_dot.append(np.hstack((wdot, dTdt)))

    phi_dot_org = np.asarray(phi_dot)
    phi_dot = pd.DataFrame(data=phi_dot_org, columns=gas.species_names + ['T'])

    # df_x_input = df_x_input.assign(H_sbr_O=df_x_input['H'] - df_x_input['O'])
    # df_x_input = df_x_input.assign(H_add_O=df_x_input['H'] + df_x_input['O'])
    # drop inert N2
    df_x_input = df_x_input_org.drop('N2', axis=1)
    # df_x_input = df_x_input.drop('dT', axis=1)
    df_y_target = df_y_target_org.drop('N2', axis=1)
    df_y_target = df_y_target_org.drop('dt', axis=1)
    # df_y_target = df_y_target.drop('T', axis=1)
    phi_dot = phi_dot.drop('N2', axis=1)

    # df_x_std = df_x_input[df_y_target['H'] > 0.005]
    # df_y_std = df_y_target[df_y_target['H'] > 0.005]
    df_x_std = df_x_input
    df_y_std = df_y_target

    # rand_sp = np.random.choice(df_x_input.index.values,200000)
    rand_sp = np.random.choice(df_x_std.index.values, 300000)
    # df_x_input = df_x_input.loc[rand_sp]
    # df_y_target = df_y_target.loc[rand_sp]
    df_x_std = df_x_std.loc[rand_sp]
    df_y_std = df_y_std.loc[rand_sp]
    phi_dot = phi_dot.loc[rand_sp]

    # create multiple nets
    nns = []
    r2s = []

    # nn_std = combustionML(df_x_input[df_y_target['H']>1e-6], df_y_target[df_y_target['H']>1e-6], 'std')
    # nn_std = combustionML(df_x_input, df_y_target, 'std')
    nn_std = combustionML(df_x_std, df_y_std, 'std')
    r2 = nn_std.run([200, 2, 0.5])
    r2s.append(r2)
    nns.append(nn_std)
    nns.append(nn_std)
    #
    #
    # # nn_log = combustionML(df_x_input[df_y_target['H'] < 1e-6], df_y_target[df_y_target['H'] < 1e-6], 'log')
    # nn_log = combustionML(df_x_std, df_y_std, 'log')
    # r2 = nn_log.run([200, 2, 0.])
    # r2s.append(r2)
    # nns.append(nn_log)
    #
    # nn_nrm = combustionML(df_x_std, df_y_std, 'nrm')
    # r2 = nn_nrm.run([200, 2, 0.5])
    # # r2s.append(r2)
    # # nns.append(nn_nrm)
    #
    #
    # # dl_react(nns, class_scaler, kmeans, 1001, 2, df_x_input_l.values[0].reshape(1,-1))
    # # cut_plot(nns, class_scaler, kmeans, 2, 'H', 0)
    #
    # cmpr, ode_o, ode_n = cmp_plot(x_columns, nns, 2, 'H', 0, 0.9)
    # cmpr, ode_o, ode_n = cmp_plot(x_columns, nns, 50, 'OH', 0, 1)
    # cmp_plot(x_columns, nns, 20, 'O', 0, 0)
    # cmp_plot(x_columns, nns, 10, 'O', 10, 1)
    # cmp_plot(x_columns, nns, 100, 'O', 10, 0)
    #
    # # c = abs(b_n[b_o != 0] - b_o[b_o != 0]) / b_o[b_o != 0]

    # %%
    phi_scale = phi_dot / nn_std.x_scaling.std.var_[:-1]

    from sklearn.decomposition import PCA

    npc = 7
    pca = PCA(n_components=npc)
    principal_components = pca.fit_transform(nn_std.x_train)
    principal_df = pd.DataFrame(data=principal_components,
                                columns=['pc' + str(x) for x in range(npc)])

    # final_df = pd.concat([principal_df, df[['target']]], axis=1)
    pca.explained_variance_ratio_.sum()
