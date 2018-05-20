import collections
import concurrent.futures
import datetime
import os
import pickle
import random
import re
import time

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

def _no_op(image):
    return image


def _shape(x):
    n, h, w, *rest = x.shape
    c = rest[0] if rest else 1
    return n, h, w, c


def _reshape(x_train, x_test):

    _, img_rows, img_cols, c = _shape(x_train)

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


def worker(preprocess_fn, dirname):
    tens_bbox = Bbox(x=2, y=0, w=15, h=30)
    ones_bbox = Bbox(x=17, y=0, w=15, h=30)
    solo_ones_bbox = Bbox(x=0, y=0, w=23, h=30)
    ult_charge_bbox = Bbox(x=625, y=590, w=30, h=30)
    shear = transform.AffineTransform(shear=0.2)
    warped_size = (28, 28)

    ult_charge = int(dirname)
    tens_digit = ult_charge // 10
    ones_digit = ult_charge % 10

    directory = os.path.join(conf.OW_ULT_CHARGE_EVAL_DATASET_DIR, dirname)

    x_valid, y_valid = [], []

    for file in os.listdir(directory):
        if file in ['.DS_Store']:
            continue

        img = mpimg.imread(os.path.join(directory, file))
        region = crop_region(img, ult_charge_bbox)
        region = transform.warp(region, inverse_map=shear)

        if tens_digit == 0:
            digit = cv2.resize(crop_region(region, solo_ones_bbox), \
                               warped_size, interpolation=cv2.INTER_LINEAR)
            digit = preprocess_fn(digit)
            x_valid.append(digit)
            y_valid.append(ones_digit)
        else:
            tens = cv2.resize(crop_region(region, tens_bbox), warped_size, interpolation=cv2.INTER_LINEAR)
            ones = cv2.resize(crop_region(region, ones_bbox), warped_size, interpolation=cv2.INTER_LINEAR)
            tens = preprocess_fn(tens)
            ones = preprocess_fn(ones)
            x_valid.append(tens)
            y_valid.append(tens_digit)
            x_valid.append(ones)
            y_valid.append(ones_digit)

    return x_valid, y_valid


def load_straight_dataset(preprocess_fn):
    """
    Parse out tens and ones digits from each image,
    and straighten each digit
    """
    tens_bbox = Bbox(x=2, y=0, w=15, h=30)
    ones_bbox = Bbox(x=17, y=0, w=15, h=30)
    solo_ones_bbox = Bbox(x=0, y=0, w=23, h=30)
    ult_charge_bbox = Bbox(x=625, y=590, w=30, h=30)
    shear = transform.AffineTransform(shear=0.2)
    warped_size = (28, 28)

    x_valid = []
    y_valid = []
    directories = [file for file in os.listdir(conf.OW_ULT_CHARGE_EVAL_DATASET_DIR)\
                   if os.path.isdir(os.path.join(conf.OW_ULT_CHARGE_EVAL_DATASET_DIR, file))]
    ult_dirs = []
    for file in directories:
        m = re.search('(^\d{1,3}$)', file)
        if m and m.groups():
            ult_dirs.append(file)

    with concurrent.futures.ThreadPoolExecutor(max_workers=conf.MAX_WORKERS) as executor:
        future_to_dirname = { executor.submit(worker, preprocess_fn, dirname) : dirname for dirname in ult_dirs }

        for future in concurrent.futures.as_completed(future_to_dirname):
            dirname = future_to_dirname[future]
            try:
                x, y = future.result()
            except Exception as e:
                print('%s generated an exception: %s' % (dirname, e))
            else:
                print('Captured %d images from %s' % (len(x), dirname))
                x_valid.extend(x)
                y_valid.extend(y)

    x_valid = np.array(x_valid)
    y_valid = np.array(y_valid)
    n, h, w, c = _shape(x_valid)
    x_valid = x_valid.reshape(n, h, w, c)
    y_valid = keras.utils.to_categorical(y_valid, conf.NUM_CLASSES)

    return x_valid, y_valid


def load_straight_grayscale_dataset(load_cached=False):
    """
    Parse out tens and ones digits from each image,
    and straighten each digit
    """

    if load_cached and os.path.exists(conf.OW_ULT_CHARGE_SHEARED_VALID_GRAYSCALE_DATASET_PKL):
        with open(conf.OW_ULT_CHARGE_SHEARED_VALID_GRAYSCALE_DATASET_PKL, 'rb') as f:
            x_valid, y_valid = pickle.load(f)
            print('Deserialized validation dataset from Pickle file.')
    else:
        x_valid, y_valid = load_straight_dataset(color.rgb2gray)

        with open(conf.OW_ULT_CHARGE_SHEARED_VALID_GRAYSCALE_DATASET_PKL, 'wb') as f:
            pickle.dump((x_valid, y_valid), f, protocol=4)
            print('Done serializing validation dataset.')

    return x_valid, y_valid


def load_straight_rgb_dataset(load_cached=True):
    """
    Parse out tens and ones digits from each image,
    and straighten each digit
    """

    if load_cached and os.path.exists(conf.OW_ULT_CHARGE_SHEARED_VALID_RGB_DATASET_PKL):
        with open(conf.OW_ULT_CHARGE_SHEARED_VALID_RGB_DATASET_PKL, 'rb') as f:
            x_valid, y_valid = pickle.load(f)
            print('Deserialized validation dataset from Pickle file.')
    else:
        x_valid, y_valid = load_straight_dataset(_no_op)

        with open(conf.OW_ULT_CHARGE_SHEARED_VALID_RGB_DATASET_PKL, 'wb') as f:
            pickle.dump((x_valid, y_valid), f, protocol=4)
            print('Done serializing validation dataset.')

    return x_valid, y_valid


def load_slanted_dataset(load_cached=False):
    """
    Parse out tens and ones digits from each image.
    """
    tens_bbox = Bbox(x=2, y=0, w=15, h=30)
    ones_bbox = Bbox(x=17, y=0, w=15, h=30)
    solo_ones_bbox = Bbox(x=0, y=0, w=23, h=30)
    ult_charge_bbox = Bbox(x=625, y=590, w=30, h=30)
    warped_size = (28, 28)

    if load_cached and os.path.exists(conf.OW_ULT_CHARGE_SLANTED_VALID_DATASET_PKL):
        with open(conf.OW_ULT_CHARGE_SLANTED_VALID_DATASET_PKL, 'rb') as f:
            x_valid, y_valid = pickle.load(f)
            print('Deserialized validation dataset from Pickle file.')
    else:
        x_valid = []
        y_valid = []
        directories = [file for file in os.listdir(conf.OW_ULT_CHARGE_EVAL_DATASET_DIR)\
                       if os.path.isdir(os.path.join(conf.OW_ULT_CHARGE_EVAL_DATASET_DIR, file))]
        ult_dirs = []
        for file in directories:
            m = re.search('(^\d{1,3}$)', file)
            if m and m.groups():
                ult_dirs.append(file)

        for dirname in ult_dirs:
            ult_charge = int(dirname)
            tens_digit = ult_charge // 10
            ones_digit = ult_charge % 10

            print('Capturing {} and {} for {}'.format(tens_digit, ones_digit, dirname))
            directory = os.path.join(conf.OW_ULT_CHARGE_EVAL_DATASET_DIR, dirname)

            for file in os.listdir(directory):
                if file in ['.DS_Store']:
                    continue

                img = mpimg.imread(os.path.join(directory, file))
                region = crop_region(img, ult_charge_bbox)

                if tens_digit == 0:
                    digit = cv2.resize(crop_region(region, solo_ones_bbox), \
                                            warped_size, interpolation=cv2.INTER_LINEAR)
                    digit = color.rgb2gray(digit)
                    x_valid.append(digit)
                    y_valid.append(ones_digit)
                else:
                    tens = cv2.resize(crop_region(region, tens_bbox), warped_size, interpolation=cv2.INTER_LINEAR)
                    ones = cv2.resize(crop_region(region, ones_bbox), warped_size, interpolation=cv2.INTER_LINEAR)
                    tens = color.rgb2gray(tens)
                    ones = color.rgb2gray(ones)
                    x_valid.append(tens)
                    y_valid.append(tens_digit)
                    x_valid.append(ones)
                    y_valid.append(ones_digit)

        with open(conf.OW_ULT_CHARGE_SLANTED_VALID_DATASET_PKL, 'wb') as f:
            pickle.dump((x_valid, y_valid), f, protocol=4)
            print('Done serializing validation dataset.')

    x_valid = np.array(x_valid)
    y_valid = np.array(y_valid)
    n, h, w, c = _shape(x_valid)
    x_valid = x_valid.reshape(n, h, w, c)
    y_valid = keras.utils.to_categorical(y_valid, conf.NUM_CLASSES)

    return x_valid, y_valid


def ow_synthetic_data_font(size):
    return ImageFont.truetype(os.path.join(os.getcwd(), "data/big_noodle_titling_oblique.ttf"), size)


def generate_digit(
        digit,
        canvas_color,
        text_color=conf.WHITE,
        text_size=32,
        dx=0,
        dy=0,
        preprocess_fn=lambda x: x):

    # Create canvas
    image = PIL.Image.new("RGB", conf.CANVAS_SIZE, canvas_color)
    canvas = ImageDraw.Draw(image)

    x, y = conf.TOP_LEFT_CORNER
    x, y = x + dx, y + dy

    # Draw text on canvas
    canvas.text(
        (x, y),
        str(digit),
        font=ow_synthetic_data_font(text_size),
        fill=text_color)

    # Prepare for consumption by neural network
    np_image = np.array(image)
    np_image = preprocess_fn(np_image)

    return np_image


def _generate_ult_meter_data(preprocess_fn):
    print('Generating training dataset...')

    canvas_colors = [(r, g, b) for r in range(0, 256, 52) for g in range(0, 256, 52) for b in range(0, 256, 52)]
    text_colors = [(r, g, b) for r in range(0, 256, 52) for g in range(0, 256, 52) for b in range(0, 256, 52)]

    x_train = []
    y_train = []

    # Make sure the training set is always consistent
    random.seed(231)

    for digit in range(10):
        print('Generating data for "%d"' % digit)
        for canvas_color in canvas_colors:
            for text_color in text_colors:
                for text_size in [32, 40]:
                    dx = random.choice([-5, 5, -2, 2, 0])
                    dy = random.choice([-2, 2, -1, 1, 0])
                    np_image = generate_digit(digit, canvas_color, text_color, text_size, \
                                              dx, dy, preprocess_fn)

                x_train.append(np_image)
                y_train.append(digit)

    x_train, y_train = np.array(x_train), np.array(y_train)
    y_train = keras.utils.to_categorical(y_train, conf.NUM_CLASSES)

    return x_train, y_train


def load_synthetic_grayscale_ow_ult_meter_data(load_cached_train=False, load_cached_test=False):
    """
    Create a dataset that's like the MNIST dataset
    for the Overwatch Ult Charge meter. The digits are upright
    instead of slanted.
    """

    if load_cached_train and os.path.exists(conf.OW_ULT_CHARGE_SYNTHETIC_GRAYSCALE_TRAIN_DATASET_PKL):
        print('Loading generated training dataset...')
        with open(conf.OW_ULT_CHARGE_SYNTHETIC_GRAYSCALE_TRAIN_DATASET_PKL, 'rb') as f:
            x_train, y_train = pickle.load(f)
    else:
        x_train, y_train = _generate_ult_meter_data(color.rgb2gray)

        with open(conf.OW_ULT_CHARGE_SYNTHETIC_GRAYSCALE_TRAIN_DATASET_PKL, 'wb') as f:
            pickle.dump((x_train, y_train), f, protocol=4)
            print('Done serializing training dataset.')

    x_test, y_test = load_straight_grayscale_dataset(load_cached_test)
    x_train, x_test, input_shape = _reshape(x_train, x_test)

    return (x_train, y_train), (x_test, y_test), input_shape


def load_synthetic_rgb_ow_ult_meter_data(load_cached_train=False, load_cached_test=False):
    """
    Create a dataset that's like the MNIST dataset
    for the Overwatch Ult Charge meter. The digits are upright
    instead of slanted.
    """

    if load_cached_train and os.path.exists(conf.OW_ULT_CHARGE_SYNTHETIC_RGB_TRAIN_DATASET_PKL):
        print('Loading generated training dataset...')
        with open(conf.OW_ULT_CHARGE_SYNTHETIC_RGB_TRAIN_DATASET_PKL, 'rb') as f:
            x_train, y_train = pickle.load(f)
    else:
        x_train, y_train = _generate_ult_meter_data(_no_op)

        with open(conf.OW_ULT_CHARGE_SYNTHETIC_RGB_TRAIN_DATASET_PKL, 'wb') as f:
            pickle.dump((x_train, y_train), f, protocol=4)
            print('Done serializing training dataset.')

    x_test, y_test = load_straight_rgb_dataset(load_cached_test)
    x_train, x_test, input_shape = _reshape(x_train, x_test)

    x_train = x_train.astype('float32')
    x_train /= 255

    return (x_train, y_train), (x_test, y_test), input_shape


def mnist_model(force_train=False, dataloader_fn=load_mnist_data, load_cached_train=False, load_cached_test=False):
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

    (x_train, y_train), (x_test, y_test), input_shape = dataloader_fn(load_cached_train, load_cached_test)

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
    model_name = os.path.join(conf.MODEL_DIR, \
                              'mnist_%d_%d_%d_%d.h5' % (timestamp.year, timestamp.month, timestamp.day, epoch))
    model.save(model_name)
    model_weight = os.path.join(conf.MODEL_DIR, \
                                'mnist_weights_%d_%d_%d_%d.h5' % (timestamp.year, timestamp.month, timestamp.day, epoch))

    model.save_weights(model_weight)

    return model


def ow_synthetic_model(force_train=False, dataloader_fn=load_synthetic_rgb_ow_ult_meter_data):
    """
    """
    if not force_train and os.path.exists(conf.MNIST_MODEL_PATH):
        print('Loading pretrained model...')
        return load_model(conf.MNIST_MODEL_PATH)

    print('Pretrained model does not exist. Training a model...')

    (x_train, y_train), (x_test, y_test), input_shape = dataloader_fn()

    model = Sequential()
    model.add(Conv2D(12, kernel_size=(5, 5), activation=conf.ACTIVATION, input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(conf.DROPOUT_RATE))
    model.add(Conv2D(32, kernel_size=(5, 5), activation=conf.ACTIVATION))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(conf.DROPOUT_RATE))
    model.add(Flatten())
    model.add(Dense(120, activation=conf.ACTIVATION))
    model.add(Dropout(conf.DROPOUT_RATE))
    model.add(Dense(84, activation=conf.ACTIVATION))
    model.add(Dropout(conf.DROPOUT_RATE))
    model.add(Dense(conf.NUM_CLASSES, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=conf.BATCH_SIZE, epochs=conf.EPOCHS, verbose=1, validation_data=(x_test, y_test))

    timestamp = datetime.datetime.today()
    epoch = int(timestamp.timestamp())
    model_name = os.path.join(conf.MODEL_DIR, \
                              'mnist_%d_%d_%d_%d.h5' % (timestamp.year, timestamp.month, timestamp.day, epoch))
    model.save(model_name)
    model_weight = os.path.join(conf.MODEL_DIR, \
                                'mnist_weights_%d_%d_%d_%d.h5' % (
                                timestamp.year, timestamp.month, timestamp.day, epoch))

    model.save_weights(model_weight)

    return model