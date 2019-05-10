#%%
import plotly.plotly as py
import numpy as np
import pandas as pd
from plotly.offline import init_notebook_mode, iplot
# from plotly.graph_objs import Contours, Histogram2dContour, Marker, Scatter
import plotly.graph_objs as go

def configure_plotly_browser_state():
  import IPython
  display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              plotly: 'https://cdn.plot.ly/plotly-1.5.1.min.js?noext',
            },
          });
        </script>
        '''))
#%%
df_load = pd.read_hdf('src/central.h5')

# df = df_load.astype('float32', copy=True)
df=df_load
print(df.shape)
old = df[0:int(df.shape[0]/3)].reset_index(drop=True)
org = df[int(df.shape[0]/3):2*int(df.shape[0]/3)].reset_index(drop=True)
new = df[int(2*df.shape[0]/3):].reset_index(drop=True)
print(org.columns)

#%%
#@title Default title text
species = "H2O2" #@param {type:"string"}

# configure_plotly_browser_state()
# init_notebook_mode(connected=False)

df_show=org.sample(frac=0.01)
error=(new.loc[df_show.index]-org.loc[df_show.index]).div(org.loc[df_show.index].dt,axis=0)

fig_db = {
    'data': [       
        {'name':'test data from table',
         'x': df_show['f'],
         'y': df_show['T'],
         'z': error[species],
        #  'z': df_show['H2'],
         'type':'scatter3d', 
        'mode': 'markers',
          'marker':{
              'size':1
          }
        }

         
    ],
    'layout': {
        'scene':{
            'xaxis': {'title':'mixture fraction'},
            'yaxis': {'title':'Temeprature'},
            'zaxis': {'title': species}
                 }
    }
}
# iplot(fig_db, filename='multiple-scatter')
iplot(fig_db)

#%%
