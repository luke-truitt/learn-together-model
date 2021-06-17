import data_pipeline as dp
import glob
import numpy as np
import pandas as pd
import os
import shutil 
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img


## Global Parameters
IMG_WIDTH=300
IMG_HEIGHT=300
IMG_DIM = (IMG_WIDTH, IMG_HEIGHT)

def process_img(img):
    img = conv_img(img)
    img = scale_img(img)
    return img

def conv_img(img):
    return img_to_array(load_img(img, target_size=IMG_DIM))

def scale_img(img):
    img_scaled = img.astype("float32")
    img_scaled /= 255 

    return img_scaled

def load_files(classes):

    class_string = '_'.join(classes)
    files = dp.build_dataset(class_string)

    X_train = []
    y_train = []
    X_val = []
    y_val = []

    for c in classes:
        train_files = files[c]["train"]
        val_files = files[c]["val"]

        train_imgs = [conv_img(img) for img in train_files]
        val_imgs = [conv_img(img) for img in val_files]

        i = 0
        while i < len(train_imgs):
            X_train.append(train_imgs[i])
            y_train.append(c)
            i = i+1

        i = 0
        while i < len(val_imgs):
            X_val.append(val_imgs[i])
            y_val.append(c)
            i = i+1

    X_train = np.array(X_train)
    X_val = np.array(X_val)

    # visualize a sample image 
    array_to_img(train_imgs[0])

    return X_train, y_train, X_val, y_val

def scale_imgs(X):

    imgs_scaled = X.astype("float32")
    imgs_scaled /= 255 

    return imgs_scaled
 
def encode_labels(y_train, y_val):
    from sklearn.preprocessing import LabelEncoder 

    le = LabelEncoder()
    le.fit(y_train)

    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)

    print(y_train[0:5], y_train_enc[0:5])
    # y_train_enc = np.asarray(y_train_enc).astype('float32').reshape((-1,1))
    # y_val_enc = np.asarray(y_val_enc).astype('float32').reshape((-1,1))
    print(y_train_enc.shape)
    return y_train_enc, y_val_enc, le

def gen_augmented_data(X_train, y_train, X_val, y_val):
    
    train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,
    width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, 
    horizontal_flip=True, fill_mode="nearest")
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow(X_train, y_train,batch_size=30)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=30)

    return train_generator, val_generator

def test_datagen(datagen, X,y):
    
    generator = datagen.flow(X, y, batch_size=1) 
    img = [next(generator) for i in range(0,5)] 
    fig, ax = plt.subplots(1,5, figsize=(16, 6))
    print("Labels:", [item[1][0] for item in img]) 
    l = [ax[i].imshow(img[i][0][0]) for i in range(0,5)]


def load_generators(classes):
    X_train, y_train, X_val, y_val = load_files(classes)
    
    X_train = scale_imgs(X_train)
    X_val = scale_imgs(X_val)
    
    y_train, y_val, _ = encode_labels(y_train, y_val)

    train_gen, val_gen = gen_augmented_data(X_train, y_train, X_val, y_val)

    return train_gen, val_gen

def get_test(classes):

    class_string = '_'.join(classes)
    files = dp.build_dataset(class_string)
    _, y_train, _, y_val = load_files(classes)

    X_test = []
    y_test = []

    for c in classes:
        test = files[c]["test"]

        test_imgs = [conv_img(img) for img in test]
        
        i = 0
        while i < len(test_imgs):
            X_test.append(test_imgs[i])
            y_test.append(c)
            i = i+1
    _, _, le = encode_labels(y_train, y_val)
    X_test = np.array(X_test)

    # visualize a sample image 
    array_to_img(test_imgs[0])

    return X_test, y_test, le
