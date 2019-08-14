import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    Input,
    concatenate,
)
from .layers import res_block


def ODENetModel(
    dim_input,
    dim_label,
    dataSet="test_default",
    batch_norm=False,
    n_neuron=64,
    branches=5,
    scale=3,
    fc=True,
):
    m_name = "{}_n{}_b{}_fc{}".format(dataSet, n_neuron, branches, fc)

    inputs = Input(shape=(dim_input,), name="input_1")
    x = Dense(n_neuron, activation="relu")(inputs)

    # less then 2 res_block, there will be variance
    x = res_block(
        x, scale, n_neuron, stage=1, block="a", bn=batch_norm, branches=branches
    )
    x = res_block(
        x, scale, n_neuron, stage=1, block="b", bn=batch_norm, branches=branches
    )
    # x = res_block(x, scale, n_neuron, stage=1, block='c', bn=batch_norm,branches=branches)

    if fc == True:
        x = Dense(100, activation="relu")(x)
    # x = Dropout(0.1)(x)
    predictions = Dense(dim_label, activation="linear", name="output_1")(x)

    model = Model(inputs=inputs, outputs=predictions)

    # model.compile(loss=loss_type, optimizer="adam", metrics=["accuracy"])

    return model
