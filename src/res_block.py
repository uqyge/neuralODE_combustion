
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras import layers
from keras import backend as K
K.set_floatx('float32')


def res_block(input_tensor, n_neuron, stage, block, d1=0.1, bn=False):
    conv_name_base = 'res_' + str(stage) + '_' + block + '_branch'
    bn_name_base = 'bn_' + str(stage) + '_' + block + '_branch'

    x = Dense(n_neuron, name=conv_name_base + '2a')(input_tensor)
    if bn:
        x = BatchNormalization(axis=-1, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    x = Dropout(d1)(x)

    x = Dense(n_neuron, name=conv_name_base + '2b')(x)
    if bn:
        x = BatchNormalization(axis=-1, name=bn_name_base + '2b')(x)
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    x = Dropout(0.)(x)
    return x