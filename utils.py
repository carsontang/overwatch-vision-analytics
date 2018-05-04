import pickle
import random
import os

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import keras

from skimage import color, data, filters, io, transform

import conf
import core


def crop_region(full_image, bbox):
    x, y, w, h = bbox
    return full_image[y:y+h, x:x+w]


def _create_grid(images, indices, n_rows=4, n_cols=4):
    h, w, *other = images[0].shape
    k = n_rows * n_cols
    display_grid = np.zeros((n_rows * h, n_cols * w))

    row_col_pairs = [(row, col) for row in range(n_rows) for col in range(n_cols)]

    for idx, (row, col) in zip(indices, row_col_pairs):
        row_start = row * h
        row_end = (row + 1) * h
        col_start = col * w
        col_end = (col + 1) * w
        display_grid[row_start:row_end, col_start:col_end] = images[idx]

    return display_grid


def create_grid(images, n_rows=4, n_cols=4):
    """
    Creates a n_rows x n_cols grid of the images corresponding
    to the first K indices (K = n_rows * n_cols). This grid itself
    is a large NumPy array.
    """
    k = n_rows * n_cols
    indices = [i for i in range(k)]
    return _create_grid(images, indices, n_rows, n_cols)


def create_rand_grid(images, n_rows=4, n_cols=4):
    """
    Creates a n_rows x n_cols grid of the images corresponding
    to K randomly chosen indices (K = n_rows * n_cols). This grid itself
    is a large NumPy array.
    """
    k = n_rows * n_cols
    indices = random.sample(range(len(images)), k)
    return _create_grid(images, indices, n_rows, n_cols)


dirname = '/Users/ctang/Documents/overwatch_object_detection/overwatch_part1_frames/smaller_dataset'

# Ult charge at 53%
start = 5474
end = 5541
fifty_three = [os.path.join(dirname, "frame_0%d.png" % i) for i in range(start, end+1)]

# Ult charge at 54%
start = 5542
end = 5574
fifty_four = [os.path.join(dirname, "frame_0%d.png" % i) for i in range(start, end+1)]


def load_straight_dataset():
    """
    Parse out tens and ones digits from each image,
    and straighten each digit
    """
    tens_bbox = core.Bbox(x=2, y=0, w=15, h=30)
    ones_bbox = core.Bbox(x=17, y=0, w=15, h=30)
    ult_charge_bbox = core.Bbox(x=625, y=590, w=30, h=30)
    shear = transform.AffineTransform(shear=0.2)
    warped_size = (28, 28)

    if os.path.exists(conf.OW_ULT_CHARGE_SHEARED_VALID_DATASET_PKL):
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
    plt.imshow(create_grid(x_valid[:200], 14, 14), cmap='gray')
    x_valid = x_valid.reshape(len(x_valid), 28, 28, 1)
    y_valid = keras.utils.to_categorical(y_valid, conf.NUM_CLASSES)

    return x_valid, y_valid