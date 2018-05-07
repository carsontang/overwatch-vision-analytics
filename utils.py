import random

import numpy as np


def crop_region(full_image, bbox):
    x, y, w, h = bbox
    return full_image[y:y+h, x:x+w]


def _create_grid(images, indices, n_rows=4, n_cols=4):
    n, h, w, *rest = images.shape
    c = rest[0] if rest else 1

    # Grayscale and RGB need differing grid dimensions
    if c > 1:
        display_grid = np.zeros((n_rows * h, n_cols * w, c))
    else:
        display_grid = np.zeros((n_rows * h, n_cols * w))

    # Uncomment the line below if you want to visualize
    # digit data with smooth contours.
    # display_grid = display_grid.astype(np.uint8)

    row_col_pairs = [(row, col) for row in range(n_rows) for col in range(n_cols)]

    for idx, (row, col) in zip(indices, row_col_pairs):
        row_start = row * h
        row_end = (row + 1) * h
        col_start = col * w
        col_end = (col + 1) * w

        if c > 1:
            display_grid[row_start:row_end, col_start:col_end, :] = images[idx]
        else:
            display_grid[row_start:row_end, col_start:col_end] = images[idx].reshape((h,w))

    return display_grid


def create_grid(images, n_rows=4, n_cols=4):
    """
    Creates a n_rows x n_cols grid of the images corresponding
    to the first K indices (K = n_rows * n_cols). If K > # of images,
    simply display all the images. This grid itself
    is a large NumPy array.
    """
    k = min(n_rows * n_cols, len(images))
    indices = [i for i in range(k)]
    return _create_grid(images, indices, n_rows, n_cols)


def create_rand_grid(images, n_rows=4, n_cols=4):
    """
    Creates a n_rows x n_cols grid of the images corresponding
    to K randomly chosen indices (K = n_rows * n_cols). If K > # of images,
    simply display all the images. This grid itself
    is a large NumPy array.
    """
    k = min(n_rows * n_cols, len(images))
    indices = random.sample(range(len(images)), k)
    return _create_grid(images, indices, n_rows, n_cols)