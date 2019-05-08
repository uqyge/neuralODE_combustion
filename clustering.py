import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import os
import glob

import pandas as pd
from deltaNets import combustionML
from boost_test import test_data, tot, create_data
from dataScaling import dataScaling
from sklearn.cluster import KMeans

if __name__ == '__main__':
    # %%
    # create_data()

    # load training
    df_x, df_y = pickle.load(open('data/x_y_org.p', 'rb'))
    # df_x, df_y = pickle.load(open('data/x_y_org_new.p', 'rb'))
    df_x_new, df_y_new = pickle.load(open('data/x_y_org_new.p', 'rb'))
    # df_x = df_x.append(df_x_new, ignore_index=True)
    # df_y = df_y.append(df_y_new, ignore_index=True)

    columns = df_x.columns
    # train_features = columns.drop(['f', 'dt'])
    train_features = columns.drop(['f', 'N2'])

    # initial conditions
    n_H2 = sorted(list(map(float, set(df_x['f']))))
    n_H2 = np.asarray(n_H2).reshape(-1, 1)

    # df_x = df_x[train_features.drop(['cp','Rho'])]
    df_x = df_x[train_features]
    df_y = df_y[train_features]

    # drop df_x == 0
    indx = (df_x != 0).all(1)
    df_x = df_x.loc[indx]
    df_y = df_y.loc[indx]

    k_scale = dataScaling()
    kmeans = KMeans(n_clusters=4, random_state=0).fit(k_scale.fit_transform(df_x, 'log_std'))
    # # %%
    # # df_x, df_y = pickle.load(open('data/x_y_org.p', 'rb'))
    k_cluster = 0
    # # idx_kmeans = kmeans.labels_ == k_cluster
    # idx_kmeans = kmeans.labels_ > -1
    # df_x_k = df_x.loc[idx_kmeans]
    # df_y_k = df_y.loc[idx_kmeans]
    df_x_k = df_x
    df_y_k = df_y


    # target
    res_dict = {'y/x': df_y_k / df_x_k,
                'y': df_y_k,
                'y-x': df_y_k - df_x_k,
                'log(y)': np.log(df_y_k + 1e-20),
                'log(y/x)':np.log((df_y_k+1e-20)/df_x_k),
                'log(y)/log(x)': np.log(df_y_k + 1e-20) / np.log(df_x_k)}

    case = 'y/x'
    # case = 'log(y)/log(x)'
    # case = 'log(y/x)'
    # case = 'y'
    # case = 'y-x'
    res = res_dict[case]

    # species = train_features
    species = train_features.drop(['dt'])
    # species_tmp = train_features.drop(['dt','cp','Rho','Hs'])

    target = pd.DataFrame(res[species], columns=species)
    # target = pd.DataFrame(res[species_tmp], columns=species_tmp)


    outlier = 5

    idx = (target < outlier).all(1)

    # idx_1 = (target < 0.999).all(1)
    # idx_2 = (target > 1.001).all(1)
    # idx_1 = target.mean(1) < 0.999
    # idx_2 = target.mean(1) > 1.001
    # idx = idx_1 | idx_2

    out_ratio = idx.sum() / target.shape[0]

    target_train = target.loc[idx]
    input_train = df_x_k.loc[idx]


#%%
    post_species = species.drop(['cp', 'Hs', 'T', 'Rho'])
    ini_T = 1501

    for sp in post_species.intersection(species):
        for n in [5]:
            input, test = test_data(ini_T, n, columns)
            input = input.drop(['N2'], axis=1)
            # input = input.drop(['cp', 'Rho'], axis=1)

            if case == 'y/x':
                test_target = test / input
            if case == 'log(y)/log(x)':
                test_target = np.log(test) / np.log(input + 1e-20)
            if case == 'y':
                test_target = test
            if case == 'y-x':
                test_target = test - input
            if case == 'log(y/x)':
                test_target = np.log(test+1e-20/input)

            f, axarr = plt.subplots(1, 2)
            axarr[0].plot(test[sp])
            # axarr[0].set_title(str(n) + '_' + sp)

            axarr[1].set_ylim(-0.005, 0.005)
            # axarr[1].set_title(str(n) + '_' + sp)
            f.suptitle(str(n) + '_' + sp +'_cluster_' + str(k_cluster))

            ax2 = axarr[1].twinx()
            kp=kmeans.predict(k_scale.transform(input))
            # ax2.plot(kp / (kmeans.n_clusters - 1) * 0.38 + 0.81, 'y', ms=2)
            ax2.plot(test_target[sp], 'bd', ms=2)
            # ax2.plot(test_target[post_species].mean(1), 'k:', ms=2)
            ax2.set_ylim(0.8, 1.2)

            plt.savefig('fig/' + str(n) + '_' + sp)
            plt.show()


    #%%
    # a=dataScaling()
    # a.scale100=1e-10
    # # sc_case = 'log_std'
    # sc_case = 'std'
    # sp = 'H2'
    # # b = a.fit_transform(target_train, sc_case)
    # # b = pd.DataFrame(data=b, columns=target_train.columns)
    # # b = a.fit_transform(input_train, sc_case)
    # # b = pd.DataFrame(data=b, columns=input_train.columns)
    # # plt.plot(target_train['OH'].sort_values().values)
    # # plt.plot(b['OH'].sort_values().values)
    # end = input_train.shape[0]
    # plt.plot(input_train[sp][:end],target_train[sp][:end],'d',ms=1)
    # plt.title(sp+'_'+sc_case)
    # plt.show()