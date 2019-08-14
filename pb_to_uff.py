#%%
import os
import shutil
import subprocess

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import plot_model
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import freeze_graph


def convertUff(dirPath, model, output_name, uff_name):
    if os.path.exists(dirPath):
        shutil.rmtree(dirPath)

    input_saved_model_dir = dirPath

    print(output_name)
    keras.experimental.export_saved_model(model, input_saved_model_dir)

    output_graph_filename = input_saved_model_dir + '/out.pb'
    output_node_names = output_name[0]
    input_binary = False
    input_saver_def_path = False
    restore_op_name = None
    filename_tensor_name = None
    clear_devices = False
    input_meta_graph = False
    checkpoint_path = None
    input_graph_filename = None
    saved_model_tags = tag_constants.SERVING

    freeze_graph.freeze_graph(input_graph_filename, input_saver_def_path,
                              input_binary, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_graph_filename, clear_devices, "", "", "",
                              input_meta_graph, input_saved_model_dir,
                              saved_model_tags)

    subprocess.call('src/pb2uff')

    if os.path.exists(uff_name):
        os.remove(uff_name)
    os.rename(dirPath + '/out.uff', uff_name)
    # shutil.rmtree(dirPath)


#%%
dirPath = './savedModel'

#%%
model_rk45 = keras.models.load_model('./rk4Model.h5')
output_name_rk45 = ['output/add_2']

model_euler = keras.models.load_model('./eulerModel.h5')
output_name_euler = ['base/inv/inv_out/BiasAdd']

# model.summary()
# plot_model(model, to_file="./outModel.png")

#%%
# convertUff(dirPath, model_rk45, output_name_rk45, 'rk45.uff')

convertUff(dirPath, model_euler, output_name_euler, 'euler.uff')

#%%
