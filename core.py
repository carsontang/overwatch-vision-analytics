import collections
import datetime
import os
import pickle
import random

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import PIL

from PIL import ImageFont, ImageDraw, Image

import keras

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model

from skimage import color, data, filters, io, transform

import conf
from utils import crop_region

Bbox = collections.namedtuple('Bbox', ['x', 'y', 'w', 'h'])


def _reshape(x_train, x_test):

    _, img_rows, img_cols, *rest = x_train.shape
    c = rest[0] if rest else 1

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], c, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], c, img_rows, img_cols)
        input_shape = (c, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, c)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, c)
        input_shape = (img_rows, img_cols, c)

    return x_train, x_test, input_shape


def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, x_test, input_shape = _reshape(x_train, x_test)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, conf.NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, conf.NUM_CLASSES)

    return (x_train, y_train), (x_test, y_test), input_shape


def load_straight_dataset(load_cached=False):
    """
    Parse out tens and ones digits from each image,
    and straighten each digit
    """
    tens_bbox = Bbox(x=2, y=0, w=15, h=30)
    ones_bbox = Bbox(x=17, y=0, w=15, h=30)
    ult_charge_bbox = Bbox(x=625, y=590, w=30, h=30)
    shear = transform.AffineTransform(shear=0.2)
    warped_size = (28, 28)

    dirname = '/Users/ctang/Documents/overwatch_object_detection/overwatch_part1_frames/smaller_dataset'

    # Ult charge at 53%
    start = 5474
    end = 5541
    fifty_three = [os.path.join(dirname, "frame_0%d.png" % i) for i in range(start, end + 1)]

    # Ult charge at 54%
    start = 5542
    end = 5574
    fifty_four = [os.path.join(dirname, "frame_0%d.png" % i) for i in range(start, end + 1)]

    if load_cached and os.path.exists(conf.OW_ULT_CHARGE_SHEARED_VALID_DATASET_PKL):
        with open(conf.OW_ULT_CHARGE_SHEARED_VALID_DATASET_PKL, 'rb') as f:
            x_valid, y_valid = pickle.load(f)
            print('Deserialized validation dataset from Pickle file.')
    else:
        x_valid = []
        y_valid = []

        for imgfile in fifty_three:
            img = mpimg.imread(imgfile)
            region = crop_region(img, ult_charge_bbox)
            region = transform.warp(region, inverse_map=shear)
            five = cv2.resize(crop_region(region, tens_bbox), warped_size, interpolation=cv2.INTER_LINEAR)
            three = cv2.resize(crop_region(region, ones_bbox), warped_size, interpolation=cv2.INTER_LINEAR)
            five = color.rgb2gray(five)
            three = color.rgb2gray(three)
            x_valid.append(five)
            y_valid.append(5)
            x_valid.append(three)
            y_valid.append(3)

        print('Done producing 5 and 3 images.')

        for imgfile in fifty_four:
            img = mpimg.imread(imgfile)
            region = crop_region(img, ult_charge_bbox)
            region = transform.warp(region, inverse_map=shear)
            five = cv2.resize(crop_region(region, tens_bbox), warped_size, interpolation=cv2.INTER_LINEAR)
            four = cv2.resize(crop_region(region, ones_bbox), warped_size, interpolation=cv2.INTER_LINEAR)
            five = color.rgb2gray(five)
            four = color.rgb2gray(four)
            x_valid.append(five)
            y_valid.append(5)
            x_valid.append(four)
            y_valid.append(4)

        print('Done producing 5 and 4 images.')

        with open(conf.OW_ULT_CHARGE_SHEARED_VALID_DATASET_PKL, 'wb') as f:
            pickle.dump((x_valid, y_valid), f)
            print('Done serializing validation dataset.')

    x_valid = np.array(x_valid)
    y_valid = np.array(y_valid)
    x_valid = x_valid.reshape(len(x_valid), 28, 28, 1)
    y_valid = keras.utils.to_categorical(y_valid, conf.NUM_CLASSES)

    return x_valid, y_valid


def load_synthetic_ow_ult_meter_data(generate=False):
    """
    Create a dataset that's like the MNIST dataset
    for the Overwatch Ult Charge meter. The digits are upright
    instead of slanted.
    """

    if not generate and os.path.exists(conf.OW_ULT_CHARGE_SYNTHETIC_TRAIN_DATASET_PKL):
        print('Loading generated training dataset...')
        with open(conf.OW_ULT_CHARGE_SYNTHETIC_TRAIN_DATASET_PKL, 'rb') as f:
            x_train, y_train = pickle.load(f)
    else:
        print('Generating training dataset...')
        canvas_size = (28, 28)
        upper_lefthand_corner = (8, -3)

        font = ImageFont.truetype(os.path.join(os.getcwd(), "data/big_noodle_titling_oblique.ttf"), 32)
        canvas_colors = [(r, g, b) for r in range(0, 256, 52) for g in range(0, 256, 52) for b in range(0, 256, 52)]
        text_colors = [(r, g, b) for r in range(0, 256, 52) for g in range(0, 256, 52) for b in range(0, 256, 52)]

        x_train = []
        y_train = []

        for digit in range(10):
            print('Generating data for "%d"' % digit)
            for canvas_color in canvas_colors:
                for text_color in text_colors:
                    image = PIL.Image.new("RGB", canvas_size, canvas_color)
                    canvas = ImageDraw.Draw(image)
                    canvas.text(upper_lefthand_corner, str(digit), font=font, fill=text_color)
                    np_image = np.array(image)
                    x_train.append(color.rgb2gray(np_image))
                    y_train.append(digit)

        x_train, y_train = np.array(x_train), np.array(y_train)
        y_train = keras.utils.to_categorical(y_train, conf.NUM_CLASSES)

        with open(conf.OW_ULT_CHARGE_SYNTHETIC_TRAIN_DATASET_PKL, 'wb') as f:
            pickle.dump((x_train, y_train), f)
            print('Done serializing training dataset.')

    """
    TODO:
    
    5/7/2018 - Make sure x_train, x_test, y_train, y_test have similar values
    to the MNIST dataset.
    
    it looks like x_train already contains values 0 <= x <= 1
    so there is no need to divide by 255. However, make sure its type
    matches the MNIST x_train type
    
    MNIST
    is 0 <= x_train <= 1? yes
    is 0 <= x_test <= 1? yes
    
    OW
    is 0 <= x_train <= 1? yes
    is 0 <= x_test <= 1? yes
    
    test code:
    np.any(x_train[100] > 1) or np.any(x_test[100] > 1)
    
    To understand the Keras training output,
    know that the loss/accuracy is calculated per batch,
    which in our case is 128 images.
    """
    x_test, y_test = load_straight_dataset()
    x_train, x_test, input_shape = _reshape(x_train, x_test)

    return (x_train, y_train), (x_test, y_test), input_shape


def mnist_model(force_train=False, dataloader_fn=load_mnist_data):
    """
    Train a simple ConvNet on the MNIST dataset,
    or loads a pretrained model if one exists. MNIST-like data
    can also be loaded by passing in a custom dataloader.
    Gets to 98.97% test accuracy after 12 epochs for MNIST.
    """
    if not force_train and os.path.exists(conf.MNIST_MODEL_PATH):
        print('Loading pretrained model...')
        return load_model(conf.MNIST_MODEL_PATH)

    print('Pretrained model does not exist. Training a model...')

    (x_train, y_train), (x_test, y_test), input_shape = dataloader_fn()

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(conf.NUM_CLASSES, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=conf.BATCH_SIZE, epochs=conf.EPOCHS, verbose=1, validation_data=(x_test, y_test))

    timestamp = datetime.datetime.today()
    epoch = int(timestamp.timestamp())
    model.save('mnist_%d_%d_%d_%d.h5' % (timestamp.year, timestamp.month, timestamp.day, epoch))
    model.save_weights('mnist_weights_%d_%d_%d_%d.h5' % (timestamp.year, timestamp.month, timestamp.day, epoch))

    return model