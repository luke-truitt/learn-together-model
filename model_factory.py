import glob
import numpy as np
import pandas as pd
import os
import shutil 
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import data_preprocessing as dp

IMG_HEIGHT = dp.IMG_HEIGHT
IMG_WIDTH = dp.IMG_WIDTH

def load_resnet(hyperparam_tuning=False, verbose=False):

    from keras.applications.resnet50 import ResNet50
    from keras.models import Model
    import keras
    resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT,IMG_WIDTH,3))
    output = resnet.layers[-1].output
    output = keras.layers.Flatten()(output)
    resnet = Model(resnet.input, output)
    if hyperparam_tuning:
        resnet.trainable = True
    set_trainable = False

    for layer in resnet.layers:
        if layer.name in ['res5c_branch2b', 'res5c_branch2c', 'activation_97']:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    else:
        for layer in resnet.layers:
            layer.trainable = False
    resnet.summary()

    return resnet
def get_vgg_model():
    from keras.applications import vgg16
    from keras.models import Model
    import keras

    vgg = vgg16.VGG16(include_top=False, weights='imagenet', 
                                        input_shape=(IMG_HEIGHT,IMG_WIDTH,3))

    output = vgg.layers[-1].output
    output = keras.layers.Flatten()(output)

    vgg_model = Model(vgg.input, output)
    vgg_model.trainable = False

    for layer in vgg_model.layers:
        layer.trainable = False

    vgg_model.summary()
    return vgg_model

def init_model(resnet, hyperparam_tuning=False, verbose=False):

    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
    from keras.models import Sequential
    from keras import optimizers

    slow_learn_rate = optimizers.RMSprop(lr=1e-5)
    fast_learn_rate = optimizers.RMSprop(lr=2e-5)

    model = Sequential()
    # vgg_model = get_vgg_model()
    # input_shape = vgg_model.output_shape[1]
    input_shape = resnet.output_shape[1]
    model.add(resnet)
    model.add(Dense(512, activation='relu', input_dim=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer=optimizers.RMSprop(lr=1e-4),
                metrics=['accuracy'])
    model.summary()

    return model

def run_fit(model, train_generator, val_generator, hyperparam_tuning=False, save_model=False, fname="", verbose=False):
    
    history = model.fit(train_generator,
                            epochs= 100,
                            validation_data=val_generator, 
                            validation_steps=100 if hyperparam_tuning else 50, 
                            verbose=1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    if save_model:
        model.save("models/{}".format(fname))

    return model, history

def train_model(classes, hyperparam_tuning=False, save_model=False, fname="test", verbose=False):
    train_gen, val_gen = dp.load_generators(classes)
    
    resnet = load_resnet(hyperparam_tuning, verbose)
    
    model = init_model(resnet, hyperparam_tuning, verbose)

    model, history = run_fit(model, train_gen, val_gen, hyperparam_tuning, save_model, fname, verbose)

    return model
    