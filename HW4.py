


from __future__ import print_function

import datetime
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from PIL import Image
import os

now = datetime.datetime.now

batch_size = 128
num_classes = 15
epochs = 5

img_rows, img_cols = 28, 28
filters = 32
pool_size = 2
kernel_size = 3

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)


def train_model(model, train, test, num_classes):
    x_train = train[0].reshape((train[0].shape[0],) + input_shape)
    x_test = test[0].reshape((test[0].shape[0],) + input_shape)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    y_train = keras.utils.to_categorical(train[1], num_classes)
    y_test = keras.utils.to_categorical(test[1], num_classes)

    model.compile(loss = 'categorical_crossentropy',
                  optimizer = 'adadelta',
                  metrics = ['accuracy'])

    t = now()
    model.fit(x_train, y_train,
              batch_size = batch_size,
              epochs = epochs,
              verbose = 1,
              validation_data = (x_test, y_test))
    print('Training time: %s' % (now() - t))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', score[1])


(x_train, y_train), (x_test, y_test) = mnist.load_data()

folder_path = '/Users/austi/OneDrive/Documents/SMU/Summer 2023/ML2/Character Images'

x_train_letters = []
y_train_letters = []

letters = ['A', 'B', 'C', 'D', 'E']
for letter in letters:
    for i in range(1, 6):
        image_path = os.path.join(folder_path, f"{letter}{i}.jpg")
        image = Image.open(image_path).convert('L')  
        image = image.resize((img_rows, img_cols))  
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis = -1)     
        x_train_letters.append(image_array)
        y_train_letters.append(letters.index(letter))  

x_train_letters = np.array(x_train_letters)
y_train_letters = np.array(y_train_letters)

x_train_letters = x_train_letters.astype('float32') / 255.0
x_train_letters = x_train_letters.reshape((x_train_letters.shape[0], ) + input_shape)

x_train_combined = np.concatenate((x_train.reshape(x_train.shape + (1, )), x_train_letters), axis = 0)
y_train_combined = np.concatenate((y_train, y_train_letters), axis = 0)
x_test_combined = x_test.reshape(x_test.shape + (1, ))
y_test_combined = y_test

feature_layers = [
    Conv2D(filters, kernel_size,
           padding = 'valid',
           input_shape = input_shape),
    Activation('relu'),
    Conv2D(filters, kernel_size),
    Activation('relu'),
    MaxPooling2D(pool_size = pool_size),
    Dropout(0.25),
    Flatten(),
]

classification_layers = [
    Dense(128),
    Activation('relu'),
    Dropout(0.5),
    Dense(num_classes),
    Activation('softmax')
]

model = Sequential(feature_layers + classification_layers)

train_model(model,
            (x_train_combined, y_train_combined),
            (x_test_combined, y_test_combined), num_classes)


# i had issues with handwritten dataset
# 
# 
# i had issues with the num of classes
# 
# 
# i had issues with letters being labeled as numbers then back to letters
# 
# 
# i had issues with tensorflow detecting repeated tracing of a function 
# 
# 
# i had so many freaking issues with shapes not being compatible 


