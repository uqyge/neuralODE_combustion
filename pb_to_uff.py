#%%
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.tools import freeze_graph
from tensorflow.python.saved_model import tag_constants
from tensorflow.keras.utils import plot_model


#%%
# model = keras.models.load_model('./postODENet.h5')

# model = keras.models.load_model('./rk4Model.h5')
# output_name = ['output/add_2']

model = keras.models.load_model('./eulerModel.h5')
output_name = ['base/inv/inv_out/BiasAdd']

model.summary()
plot_model(model, to_file="./outModel.png")

#%%
!rm -rf ./savedModel
input_saved_model_dir = './savedModel'
keras.experimental.export_saved_model(model, input_saved_model_dir)

#%%
# print(model.layers[-1].outputs[0].op)
# output_name = [t.op.name for t in model.outputs]
# output_name = [t.op.inputs[0].op.name for t in model.outputs] 


# output_name = ['sequential/dense/BiasAdd']
# output_name = ['sequential_2/BiasAdd']
print(output_name)

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

#%%
!./src/pb2uff

#%%
