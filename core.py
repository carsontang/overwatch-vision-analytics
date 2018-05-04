import collections
import datetime
import os

import keras

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model

import conf

Bbox = collections.namedtuple('Bbox', ['x', 'y', 'w', 'h'])


def load_mnist_data():
	img_rows, img_cols = 28, 28

	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	if K.image_data_format() == 'channels_first':
		x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
		x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
		input_shape = (1, img_rows, img_cols)
	else:
		x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
		x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
		input_shape = (img_rows, img_cols, 1)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255

	y_train = keras.utils.to_categorical(y_train, conf.NUM_CLASSES)
	y_test = keras.utils.to_categorical(y_test, conf.NUM_CLASSES)

	return (x_train, y_train), (x_test, y_test), input_shape


def mnist_model(force_train=False):
	"""
	Train a simple ConvNet on the MNIST dataset,
	or loads a pretrained model if one exists.
	Gets to 98.97% test accuracy after 12 epochs
	"""
	if not force_train and os.path.exists(conf.MNIST_MODEL_PATH):
		print('Loading pretrained model...')
		return load_model(conf.MNIST_MODEL_PATH)

	print('Pretrained model does not exist. Training a model...')

	(x_train, y_train), (x_test, y_test), input_shape = load_mnist_data()

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
