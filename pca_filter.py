# %%
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pickle

from reactor_ode_delta import ignite_post, data_gen_f
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from dataScaling import LogScaler, AtanScaler, NoScaler
from sklearn.decomposition import PCA
from boost_test import test_data,sp_plot_gpu_mask


if __name__ == '__main__':
    # generate data
    # create_data()

    # load training
    df_x, df_y = pickle.load(open('data/x_y_org.p', 'rb'))
    columns = df_x.columns
    # df_x = df_x.drop('N2', axis=1)
    f = set(df_x['f'])
    f = np.asarray(sorted(list(map(float, f)))).reshape(-1, 1)
    df_x = df_x.drop(['f','dt'], axis=1)
    df_y = df_y.drop('N2', axis=1)

    indx = (df_x != 0).all(1)
    df_x = df_x.loc[indx]
    df_y = df_y.loc[indx]
#%%
    pca= PCA(n_components=3) # pca for 3D visualization

    # train data
    # df_x_s=df_x.sample(frac=0.1) # take 10% data
    df_x_s = df_x.copy()

    scaler_pca = NoScaler()
    train=pca.fit_transform(scaler_pca.transform(df_x_s))
    print('variance:',sum(pca.explained_variance_ratio_))
    train = np.hstack((train,0*np.ones((train.shape[0],1)))) # add a label

    delta = 0.01
    cube_train = np.round(train / delta)[:,:3]
    skt=set(tuple(i)for i in cube_train.tolist())
    train_ijk=[tuple(i) for i in cube_train.tolist()]

    len(skt)

    df_pca = pd.DataFrame(data=train[:, 0:4], columns=['x', 'y', 'z', 'label'])
    df_pca = df_pca.reindex(df_x.index)
    df_pca['ijk'] = train_ijk

    df_pca_filtered=df_pca.drop_duplicates('ijk',keep='first')

    df_x=df_x_s.loc[df_pca_filtered.index]
    df_y=df_y.loc[df_pca_filtered.index]


    #%%
    res = (df_y-df_x)/df_x

    #%%
    df_x['C'] = df_x['H2'] + df_x['H2O']

    df_x['tot:O'] = 2 * df_x['O2'] + df_x['OH'] + df_x['O'] + df_x['H2O'] \
                    + 2 * df_x['HO2'] + 2 * df_x['H2O2']

    df_x['tot:H'] = 2 * df_x['H2'] + df_x['H'] + df_x['OH'] + 2 * df_x['H2O'] \
                    + df_x['HO2'] + 2 * df_x['H2O2']

    # %%
    # mask = df_x['O2'] < 0.01
    # mask_train = (df_x['O2'] < 0.01) | (df_x['H2'] < 0.01)
    mask_train = df_x['HO2'] < 1
    bstReg_gpu = {}
    scalers = {}
    # species = ['O2', 'O', 'H2O', 'H2O2']
    species = ['O2', 'O']

    # species = ['H2','OH','O2','O','H2O']
    # species = columns.drop(['dt','N2','f'])

    for state in {'b:', 'u:'}:
        mask_train = ~mask_train

        if mask_train.any():
            df_x_masked = df_x.loc[mask_train]
            df_y_masked = res.loc[mask_train]
            # df_y_masked[df_y_masked>0.2]=np.nan
            X_train, X_test, y_train, y_test = train_test_split(df_x_masked, df_y_masked,
                                                                test_size=.1, random_state=42)

            for sp in species:
                # scaler = LogScaler()
                # scaler = AtanScaler()
                # scaler = NoScaler()
                # scaler = MinMaxScaler()
                scaler = MaxAbsScaler()
                # scaler = StandardScaler()

                outlier = 100
                # target_train = np.log(y_train[sp])
                target_train = scaler.fit_transform(y_train[sp].values.reshape(-1, 1))
                # dtrain = xgb.DMatrix(X_train, label=target_train)
                dtrain = xgb.DMatrix(X_train[target_train < outlier], label=target_train[target_train < outlier])

                # target_test = np.log(y_test[sp])
                target_test = scaler.transform(y_test[sp].values.reshape(-1, 1))
                # dtest = xgb.DMatrix(X_test, label=target_test)
                dtest = xgb.DMatrix(X_test[target_test < outlier], label=target_test[target_test < outlier])
                param = {
                    'max_depth': 10,
                    'gamma': 0.3,
                    'eta': 0.25,
                    'silent': 1,
                    'eval_metric': 'mae',
                    'predictor': 'gpu_predictor',
                    'objective': 'gpu:reg:linear'
                }

                num_round = 100
                bst = xgb.train(param, dtrain, num_round,
                                evals=[(dtest, 'test')], early_stopping_rounds=10)

                bstReg_gpu[state + sp] = bst
                scalers[state + sp] = scaler
                # print(sp + ':', r2_score(np.exp(bst.predict(dtest)), target_test))
                print(sp + ':', r2_score(np.exp(bst.predict(dtest)), target_test[target_test < outlier]))

    xgb.plot_importance(bst)
    plt.show()

    # load test
    # %%

    for sp_test in species:
        # for n in [.5, 1.4, 2.6, 5, 10, 13, 25]:
        for n in [1,25]:
            ode_o, ode_n = test_data(1501, n, columns)
            ode_o = ode_o.drop('dt',axis=1)
            ode_o['C'] = ode_o['H2'] + ode_o['H2O']
            ode_o['tot:O'] = 2 * ode_o['O2'] + ode_o['OH'] + ode_o['O'] + ode_o['H2O'] \
                             + 2 * ode_o['HO2'] + 2 * ode_o['H2O2']
            ode_o['tot:H'] = 2 * ode_o['H2'] + ode_o['H'] + ode_o['OH'] + 2 * ode_o['H2O'] \
                             + ode_o['HO2'] + 2 * ode_o['H2O2']
            # mask_pred = (ode_o['H2'] < 0.01) | (ode_o['O2'] < 0.01)
            mask_pred = ode_o['HO2'] < 1
            # mask_pred = ~mask_pred
            sp_plot_gpu_mask(n, sp_test, bstReg_gpu, scalers, ode_o, ode_n, mask_pred)
            plt.show()




