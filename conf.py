import os

MNIST_MODEL_DIR = 'model'
MNIST_MODEL_PATH = os.path.join(MNIST_MODEL_DIR, 'mnist_2018_03_17.h5')

""" Train/eval configs"""
BATCH_SIZE = 128
EPOCHS = 12
NUM_CLASSES = 10

DATA_DIR = 'data'
OW_ULT_CHARGE_EVAL_DATASET_DIR = '/Users/ctang/Documents/overwatch_object_detection/videos/01_26_2018_22_41_04_02'
OW_ULT_CHARGE_EVAL_DATASET_DIR2 = '/Users/ctang/Documents/overwatch_object_detection/overwatch_part1_frames/ult_charge'
OW_ULT_CHARGE_SYNTHETIC_TRAIN_DATASET_PKL = os.path.join(DATA_DIR, 'synthetic_train_dataset.pkl')
OW_ULT_CHARGE_VALID_DATASET_PKL = os.path.join(DATA_DIR, 'validation_dataset.pkl')
OW_ULT_CHARGE_SHEARED_VALID_DATASET_PKL = os.path.join(DATA_DIR, 'valid_straight_dataset.pkl')