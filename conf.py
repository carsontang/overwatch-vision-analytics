import os

MNIST_MODEL_DIR = 'model'
MNIST_MODEL_PATH = os.path.join(MNIST_MODEL_DIR, 'mnist_2018_03_17.h5')

""" Train/eval configs"""
BATCH_SIZE = 128
EPOCHS = 12
NUM_CLASSES = 10

OW_ULT_CHARGE_VALID_DATASET_PKL = 'validation_dataset.pkl'
OW_ULT_CHARGE_SHEARED_VALID_DATASET_PKL = 'valid_straight_dataset.pkl'