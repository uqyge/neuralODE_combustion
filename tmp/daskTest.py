import dask.dataframe as dd
import pandas as pd
import time


s = time.time()

# # df_load = pd.read_hdf('data/merged.h5')
# df_load = pd.read_hdf('src/central.h5')

# df = df_load.astype('float32', copy=True)
# # df=df_load
# print(df.shape)
# # define the labels
# labels = df.columns.drop(['dt', 'f', 'Hs', 'cp'])
# # labels=['HO2']

# # input_features=df.columns.drop(['f'])
# input_features = labels

# old = df[0:int(df.shape[0]/3)].reset_index(drop=True)
# org = df[int(df.shape[0]/3):2*int(df.shape[0]/3)].reset_index(drop=True)
# new = df[int(2*df.shape[0]/3):].reset_index(drop=True)

# old.to_hdf('tmp.h5', key='old', format='table')
# org.to_hdf('tmp.h5', key='org', format='table')
# new.to_hdf('tmp.h5', key='new', format='table')


# %%
ddOld = dd.read_hdf('tmp.h5', key='old')
ddOrg = dd.read_hdf('tmp.h5', key='org')
ddNew = dd.read_hdf('tmp.h5', key='new')
ddOld.describe().compute()

idx_dt = (ddOld.dt < 5e-7) & (ddOld.dt > 5e-9)
ddOld = ddOld[idx_dt]
ddOrg = ddOrg[idx_dt]
ddNew = ddNew[idx_dt]
# ddOld.describe().compute()

idx = (ddOrg > 0).all(1)
ddOld = ddOld[idx]
ddOrg = ddOrg[idx]
ddNew = ddNew[idx]

labels = ddOld.columns.drop(['dt', 'f', 'Hs', 'cp'])
idx_f = abs((ddNew[labels]-ddOld[labels]).div(ddOld.dt+ddOrg.dt, axis=0)).max(1) > 0.05
ddOld = ddOld[idx_f]
ddOrg = ddOrg[idx_f]
ddNew = ddNew[idx_f]

# len(ddOld)


# %%
# old = pd.read_hdf('tmp.h5', key='old')
# org = pd.read_hdf('tmp.h5', key='org')
# new = pd.read_hdf('tmp.h5', key='new')

# idx_dt = (org['dt'] < 5e-7) & (org['dt'] > 5e-9)
# # print(sum(idx_dt))
# old = old[idx_dt]
# org = org[idx_dt]
# new = new[idx_dt]

# idx = (org > 0).all(1)
# # print(sum(idx))
# old = old[idx]
# org = org[idx]
# new = new[idx]

# # idx_f = ((new/org) < 5).all(1)
# labels = org.columns.drop(['dt', 'f', 'Hs', 'cp'])
# idx_f = abs((new[labels]-org[labels]).div(org.dt, axis=0)).max(1) > 0.05
# print(sum(idx_f))
# old = old[idx_f]
# org = org[idx_f]
# new = new[idx_f]


e = time.time()
len(ddOld)
print(e-s)


#%%
