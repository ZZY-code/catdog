from keras.layers import *
import os
import os
import tensorflow as tf
from tensorflow.python.framework import graph_util, graph_io
from tensorflow.python.tools import import_pb_to_tensorboard
from keras.models import load_model
from keras.models import Model
from keras.models import Sequential

model_path = "D:/Code/python/Data/model/train/RGB/epoch50/model.h5"
weight_path= "./weigth50.h5"

RUN_MAX = True
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
CHANNEL_SIZE=3
BATCH_SIZE=15


def keras_to_tensorflow(keras_model, output_dir, model_name, out_prefix="output_", log_tensorboard=True):
    if os.path.exists(output_dir) == False:
        os.mkdir(output_dir)

    out_nodes = []

    for i in range(len(keras_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(keras_model.output[i], out_prefix + str(i + 1))

    sess = K.get_session()

    init_graph = sess.graph.as_graph_def()
    main_graph = graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)
    graph_io.write_graph(main_graph, output_dir, name=model_name, as_text=False)

    if log_tensorboard:
        import_pb_to_tensorboard.import_to_tensorboard(
            os.path.join(output_dir, model_name), output_dir)


# def squeezenet_fire_module(input,
#
#     input_channel_small = 16,input_channel_large = 64):
#
#     channel_axis=3
#
#     input=Conv2D(input_channel_small,(1, 1),padding = "valid")(input)
#     input=Activation("relu")(input)
#
#     input_branch_1=Conv2D(input_channel_large,(1, 1),padding = "valid")(input)
#     input_branch_1=Activation("relu")(input_branch_1)
#
#     input_branch_2=Conv2D(input_channel_large,(3,3),padding = "same")(input)
#     input_branch_2=Activation("relu")(input_branch_2)
#
#     input=concatenate([input_branch_1,input_branch_2], axis = channel_axis)
#
#     return input



def SqueezeNet(input_shape=(128, 128, 3)):
    image_input=Input(shape=input_shape)

    network=Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNEL_SIZE))(image_input)
    network=MaxPooling2D(pool_size=(2, 2))(network)

    network = Conv2D(64, (3, 3), activation='relu')(network)
    network = MaxPooling2D(pool_size=(2, 2))(network)

    network = Conv2D(128, (3, 3), activation='relu')(network)
    network = MaxPooling2D(pool_size=(2, 2))(network)
    network = Flatten()(network)

    network = Dense(512, activation='relu')(network)
    network = Dense(2, activation='softmax',name="output")(network)

    input_image=image_input
    model=Model(inputs=input_image,outputs =network)

    return model

keras_model = SqueezeNet()

#keras_model.load_model(model_path)
keras_model.load_weights(weight_path)
output_dir = os.path.join(os.getcwd(), "checkpoint")

keras_to_tensorflow(keras_model, output_dir=output_dir, model_name="model.pb")

print("MODEL SAVED")
