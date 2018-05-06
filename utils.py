import random

import numpy as np


def crop_region(full_image, bbox):
    x, y, w, h = bbox
    return full_image[y:y+h, x:x+w]


def _create_grid(images, indices, n_rows=4, n_cols=4):
    h, w, *other = images[0].shape
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