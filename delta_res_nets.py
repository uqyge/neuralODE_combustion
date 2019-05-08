import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from boost_test import test_data
from src.dataScaling import dataScaling
from src.deltaNets import combustionML


def clear_hist():
    # clean start
    files = glob.glob('./tmp/history/*.hdf5') \
            + glob.glob('./tmp/*.hdf5')
    for file in files:
        os.remove(file)


if __name__ == '__main__':
    # %%
    # # create data
    # create_data()

    # # clean start
    clear_hist()

    # load training
    # df_x, df_y = pickle.load(open('data/x_y_org.p', 'rb'))
    # df_x, df_y = pickle.load(open('data/x_y_org_pca_reduced.p', 'rb'))
    # df_x, df_y = pickle.load(open('data/x_y_org_new.p', 'rb'))

    df = pd.read_hdf('./data/merged.h5')
    df_x = df[0:int(df.shape[0] / 2)]
    df_y = df[int(df.shape[0] / 2):]

    # df_x_new, df_y_new = pickle.load(open('data/x_y_org_new.p', 'rb'))
    # df_x = df_x.append(df_x_new, ignore_index=True)
    # df_y = df_y.append(df_y_new, ignore_index=True)

    # initial conditions
    n_H2 = sorted(list(map(float, set(df_x['f']))))
    n_H2 = np.asarray(n_H2).reshape(-1, 1)

    columns = df_x.columns
    # train_features = columns.drop(['f', 'dt'])
    train_features = columns.drop(['f', 'N2'])

    df_x = df_x[train_features]
    df_y = df_y[train_features]

    # drop df_x == 0
    indx = (df_x != 0).all(1)
    df_x = df_x.loc[indx]
    df_y = df_y.loc[indx]

    k_cluster = 3

    df_x_k = df_x
    df_y_k = df_y

    # target
    res_dict = {'y/x': df_y_k / df_x_k,
                'y': df_y_k,
                'y-x': df_y_k - df_x_k,
                'log(y)': np.log(df_y_k + 1e-20),
                'log(y/x)': np.log((df_y_k + 1e-20) / df_x_k),
                # 'log(y)/log(x)': np.log(df_y_k + 1e-20) / np.log(df_x_k)
                'log(y)/log(x)': np.log(df_y_k) / np.log(df_x_k)
                }

    # case = 'y/x'
    case = 'log(y)/log(x)'
    # case = 'log(y/x)'
    # case = 'y'
    # case = 'y-x'

    res = res_dict[case]
    # species = train_features
    species = train_features.drop(['dt'])
    species_tmp = train_features.drop(['dt', 'cp', 'Rho', 'Hs'])

    target = pd.DataFrame(res[species], columns=species)
    # target = pd.DataFrame(res[species_tmp], columns=species_tmp)

    # outlier = 1.1
    outlier = 5e10

    idx = (target < outlier).all(1)

    # idx_1 = (target < 0.999).all(1)
    # idx_2 = (target > 1.001).all(1)
    # idx_1 = target.mean(1) < 0.999
    # idx_2 = target.mean(1) > 1.001
    # idx = idx_1 | idx_2

    out_ratio = idx.sum() / target.shape[0]

    input_train = df_x_k.loc[idx]
    target_train = target.loc[idx]

    # %%
    # model formulate
    # nn_std = combustionML(df_x, target, {'x': 'log_std', 'y': 'log_std'})
    # nn_std = combustionML(df_x, target, {'x': 'std2', 'y': 'std2'})
    nn_std = combustionML(input_train, target_train, {'x': 'log_std', 'y': 'log_std'})
    # nn_std.ensemble_num = 5
    r2 = nn_std.run([200, 4, 0., 500])

    nn_std.plt_loss()

    # %%
    # test interpolation
    batch_predict = 1024 * 256
    ensemble_mode = True
    # ensemble_mode = False
    post_species = species.drop(['cp', 'Hs', 'T', 'Rho'])

    ini_T = 1501
    for sp in post_species.intersection(species):
        for n in [3]:
            input, test = test_data(ini_T, n, columns)
            input = input[train_features]

            if ensemble_mode is True:
                pred = pd.DataFrame(nn_std.inference_ensemble(input, batch_size=batch_predict), columns=target.columns)
            else:
                pred = pd.DataFrame(nn_std.inference(input), columns=target.columns)

            model_pred = pred

            if case == 'y/x':
                pred = input * pred
                test_target = test / input
            if case == 'log(y)/log(x)':
                pred = np.exp(np.log(input) * pred)
                test_target = np.log(test) / np.log(input)
            if case == 'y':
                pred = pred
                test_target = test
            if case == 'y-x':
                pred = input + pred
                test_target = test - input
            if case == 'log(y/x)':
                pred = input * np.exp(pred) - 1e-20
                test_target = np.log(test + 1e-20 / input)

            f, axarr = plt.subplots(1, 2)
            f.suptitle(str(n) + '_' + sp + '_cluster_' + str(k_cluster))

            axarr[0].plot(test[sp])
            axarr[0].plot(pred[sp], 'rd', ms=2)
            # axarr[0].set_title(str(n) + '_' + sp)

            # plot accuracy
            axarr[1].plot((test[sp] - pred[sp]) / test[sp], 'kd', ms=1)
            axarr[1].set_ylim(-0.005, 0.005)

            ax2 = axarr[1].twinx()
            # ax2.plot(test_target.mean(1), 'k:', ms=2)
            # the first few may be off( good to check)
            ax2.plot(test_target[sp][1:], 'bd', ms=2)
            ax2.plot(model_pred[sp][1:], 'rd', ms=2)
            # ax2.set_ylim(0.8, 1.2)
            # kp = kmeans.predict(k_scale.transform(input))
            # ax2.plot(kp/(kmeans.n_clusters-1)*0.38+0.81,'y',ms=2)
            plt.savefig('fig/' + str(n) + '_' + sp)
            plt.show()

    # %%
    # integration
    for sp in post_species.intersection(species):
        for n in [3]:
            input, test = test_data(ini_T, n, columns)
            input = input[train_features]
            test = test[train_features]

            init = 0
            input_model = input.values[init].reshape(1, -1)
            test_model = test.values[init].reshape(1, -1)
            pred_acc = []
            test = test[init:].reset_index(drop=True)

            for i in range(input.shape[0] - init):

                if ensemble_mode is True:
                    pred_model = nn_std.inference_ensemble(input_model, batch_size=batch_predict)
                else:
                    pred_model = nn_std.inference(input_model)

                if case == 'y/x':
                    input_model[0][:-1] = input_model[0][:-1] * pred_model
                if case == 'log(y)/log(x)':
                    input_model[0][:-1] = np.exp(np.log(input_model[0][:-1]) * pred_model)
                if case == 'y':
                    input_model[0][:-1] = pred_model
                if case == 'y-x':
                    input_model[0][:-1] = pred_model + input_model[0][:-1]
                if case == 'log(y/x)':
                    input_model[0][:-1] = np.exp(pred_model) * input_model[0][:-1] - 1e-20

                # dt = 1e-6
                input_model[0][-1] = 1e-6

                pred_acc.extend(input_model.tolist())

            pred_acc = np.asarray(pred_acc)
            pred_acc = pd.DataFrame(pred_acc, columns=train_features)

            f, axarr = plt.subplots(1, 2)
            f.suptitle('Intigration: ' + str(n) + '_' + sp)
            axarr[0].plot(test[sp], 'bd', ms=2)
            axarr[0].plot(pred_acc[sp], 'rd', ms=2)
            # axarr[0].set_ylim(0, test[sp].max())

            # plot accuracy
            axarr[1].plot(abs(test[sp] - pred_acc[sp]) / test[sp], 'kd', ms=1)
            # axarr[1].set_ylim(-0.005, 0.005)

            plt.savefig('fig/acc_' + str(n) + '_' + sp)
            plt.show()

    # %%
    a = dataScaling()
    a.scale100 = 1e-10
    # sc_case = 'log_std'
    sc_case = 'std'
    sp = 'OH'
    # b = a.fit_transform(target_train, sc_case)
    # b = pd.DataFrame(data=b, columns=target_train.columns)
    b = a.fit_transform(input_train, sc_case)
    b = pd.DataFrame(data=b, columns=df_x.columns)
    # plt.plot(target_train['OH'].sort_values().values)
    # plt.plot(b['OH'].sort_values().values)
    end = input_train.shape[0]
    plt.plot(input_train[sp][:end], target_train[sp][:end], 'd', ms=1)
    plt.title(sp + '_' + sc_case)
    plt.show()
