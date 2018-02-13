
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv1D
from keras.layers import Activation
from keras.layers import MaxPooling1D
from keras.layers import GlobalAveragePooling1D
from keras.layers import Dropout
from keras.layers import Dense

import tensorflow as tf

from keras import Model
from keras import backend as K

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io


best_weights_path = 'basic_cnn_1d.h5'
input_shape = (5000, 1)

num_output = 1

output_node_names_of_final_network = 'output_node'
output_graph_name = 'basic_cnn_1d.pb'

target_names = ['Ae. aegypti', 'Ae. albopictus', 'An. gambiae', 'An. arabiensis', 'C. pipiens', 'C. quinquefasciatus']


def basic_cnn_1d(cols, channels, num_classes):
    inputs = Input(shape = (cols, channels))

    x = BatchNormalization() (inputs)
    x = Conv1D(16, kernel_size = 3, padding = 'same') (x)
    x = BatchNormalization() (x)
    x = Activation('elu') (x)
    x = MaxPooling1D((2)) (x)
    x = Conv1D(32, kernel_size = 3, padding = 'same') (x)
    x = BatchNormalization() (x)
    x = Activation('elu') (x)
    x = MaxPooling1D((2)) (x)
    x = Conv1D(64, kernel_size = 3, padding = 'same') (x)
    x = BatchNormalization() (x)
    x = Activation('elu') (x)
    x = MaxPooling1D((2)) (x)
    x = Conv1D(128, kernel_size = 3, padding = 'same') (x)
    x = BatchNormalization() (x)
    x = Activation('elu') (x)
    x = MaxPooling1D((2)) (x)
    x = Conv1D(256, kernel_size = 3, padding = 'same') (x)
    x = BatchNormalization() (x)
    x = Activation('elu') (x)
    x = MaxPooling1D((2)) (x)

    x = GlobalAveragePooling1D() (x)

    x = Dropout(0.5) (x)

    x = Dense(num_classes) (x)
    outputs = Activation('softmax') (x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return model


K.set_learning_phase(0)

model = basic_cnn_1d(input_shape[0], input_shape[1], len(target_names))

model.load_weights(best_weights_path)

pred = [None] * num_output
pred_node_names = [None] * num_output

for i in range(num_output):
    pred_node_names[i] = output_node_names_of_final_network
    pred[i] = tf.identity(model.output[i], name = pred_node_names[i])

sess = K.get_session()

constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
graph_io.write_graph(constant_graph, './', output_graph_name, as_text = False)
