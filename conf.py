import os

MNIST_MODEL_DIR = 'model'
MNIST_MODEL_PATH = os.path.join(MNIST_MODEL_DIR, 'mnist_2018_03_17.h5')

""" Train/eval configs"""
BATCH_SIZE = 128
EPOCHS = 12
NUM_CLASSES = 10

DATA_DIR = 'data'
OW_ULT_CHARGE_SYNTHETIC_TRAIN_DATASET_PKL = os.path.join(DATA_DIR, 'synthetic_train_dataset.pkl')
OW_ULT_CHARGE_VALID_DATASET_PKL = 'validation_dataset.pkl'
OW_ULT_CHARGE_SHEARED_VALID_DATASET_PKL = os.path.join(DATA_DIR, 'valid_straight_dataset.pkl')