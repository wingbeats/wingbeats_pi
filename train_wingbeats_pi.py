
## Dataset can be found at "https://www.kaggle.com/potamitis/wingbeats"

from __future__ import division
import os, math
import numpy as np
seed = 2018
np.random.seed(seed)

import soundfile as sf

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv1D
from keras.layers import Activation
from keras.layers import MaxPooling1D
from keras.layers import GlobalAveragePooling1D
from keras.layers import Dropout
from keras.layers import Dense

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger

from keras import Model
from keras import backend as K

from keras.utils import np_utils


model_name = 'basic_cnn_1d'
best_weights_path = model_name + '.h5'
log_path = model_name + '.log'
monitor = 'val_acc'
batch_size = 32
epochs = 100
es_patience = 7
rlr_patience = 3

SR = 8000
input_shape = (5000, 1)


target_names = ['Ae. aegypti', 'Ae. albopictus', 'An. gambiae', 'An. arabiensis', 'C. pipiens', 'C. quinquefasciatus']

X_names = []
y = []
target_count = []

for i, target in enumerate(target_names):
    target_count.append(0)
    path = './Wingbeats/' + target + '/'
    for [root, dirs, files] in os.walk(path, topdown = False):
        for filename in files:
            name,ext = os.path.splitext(filename)
            if ext == '.wav':
                name = os.path.join(root, filename)
                y.append(i)
                X_names.append(name)
                target_count[i]+=1
                # if target_count[i] > 20000:
                #     break
    print (target, '#recs = ', target_count[i])

print ('total #recs = ', len(y))

X_names, y = shuffle(X_names, y, random_state = seed)
X_train, X_test, y_train, y_test = train_test_split(X_names, y, stratify = y, test_size = 0.20, random_state = seed)

print ('train #recs = ', len(X_train))
print ('test #recs = ', len(X_test))


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


def random_data_shift(data):
    data = np.roll(data, int(round(np.random.uniform(-(len(data) / 4), (len(data) / 4)))))
    
    return data

def train_generator():
    while True:
        for start in range(0, len(X_train), batch_size):
            x_batch = []
            y_batch = []
            
            end = min(start + batch_size, len(X_train))
            train_batch = X_train[start:end]
            labels_batch = y_train[start:end]
            
            for i in range(len(train_batch)):        
                data, rate = sf.read(train_batch[i])

                data = random_data_shift(data)
                
                data = data / max(data)
                
                data = np.expand_dims(data, axis = -1)

                x_batch.append(data)
                y_batch.append(labels_batch[i])

            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)
            
            y_batch = np_utils.to_categorical(y_batch, len(target_names))
            
            yield x_batch, y_batch

def valid_generator():
    while True:
        for start in range(0, len(X_test), batch_size):
            x_batch = []
            y_batch = []
            
            end = min(start + batch_size, len(X_test))
            test_batch = X_test[start:end]
            labels_batch = y_test[start:end]
            
            for i in range(len(test_batch)):
                data, rate = sf.read(test_batch[i])

                data = data / max(data)

                data = np.expand_dims(data, axis = -1)

                x_batch.append(data)
                y_batch.append(labels_batch[i])

            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)
            
            y_batch = np_utils.to_categorical(y_batch, len(target_names))
            
            yield x_batch, y_batch


model = basic_cnn_1d(input_shape[0], input_shape[1], len(target_names))

callbacks_list = [ModelCheckpoint(monitor = monitor,
                                filepath = best_weights_path, 
                                save_best_only = True, 
                                save_weights_only = True,
                                verbose = 1), 
                    EarlyStopping(monitor = monitor,
                                patience = es_patience, 
                                verbose = 1),
                    ReduceLROnPlateau(monitor = monitor,
                                factor = 0.1, 
                                patience = rlr_patience, 
                                verbose = 1),
                    CSVLogger(filename = log_path)]

model.fit_generator(train_generator(),
    steps_per_epoch = int(math.ceil(float(len(X_train)) / float(batch_size))),
    validation_data = valid_generator(),
    validation_steps = int(math.ceil(float(len(X_test)) / float(batch_size))),
    epochs = epochs,
    callbacks = callbacks_list,
    shuffle = False)

model.load_weights(best_weights_path)

loss, acc = model.evaluate_generator(valid_generator(),
        steps = int(math.ceil(float(len(X_test)) / float(batch_size))))

print('loss:', loss)
print('Test accuracy:', acc)
