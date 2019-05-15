import keras
import numpy as np
# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from tflearn.data_utils import image_preloader

# Global parameters
batch_size = 64
num_classes = 10
epochs = 12
num_classes = 27
img_size = 56
train_dir = 'figures/train'
test_dir = 'figures/test'
input_shape = (img_size, img_size, 1)


def load_data(train=True):
    if train:
        target_path = train_dir
    else:
        target_path = test_dir
    x, y = image_preloader(target_path=target_path,
                           image_shape=(img_size, img_size),
                           mode='folder',
                           categorical_labels=True,
                           normalize=True,
                           grayscale=False)

    x = np.asarray(x[:])
    y = np.asarray(y[:])

    return x, y


x_train, y_train = load_data()
x_test, y_test = load_data(train=False)
x_train = x_train.reshape(x_train.shape[0], img_size, img_size, 1)
x_test = x_test.reshape(x_test.shape[0], img_size, img_size, 1)


# Initialize model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# Data augmentation
gen = ImageDataGenerator(rotation_range=8,
                         width_shift_range=0.08,
                         shear_range=0.3,
                         height_shift_range=0.08,
                         zoom_range=0.08)
test_gen = ImageDataGenerator()

train_generator = gen.flow(x_train, y_train, batch_size=batch_size)
test_generator = test_gen.flow(x_test, y_test, batch_size=batch_size)

# Run model
model.fit_generator(train_generator,
                    steps_per_epoch=60000 // batch_size,
                    epochs=5,
                    validation_data=test_generator,
                    validation_steps=10000 // batch_size)


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
