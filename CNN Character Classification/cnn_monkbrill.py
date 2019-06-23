import keras
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from tflearn.data_utils import image_preloader
from keras.callbacks import ModelCheckpoint
from PIL import Image
from models import *
# from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

# Global parameters
batch_size = 32
num_epochs = 50
num_classes = 27
img_width = 32
img_height = 48
train_dir = 'correct_figures_monkbrill/train_cor'
# test_dir = 'figures_monkbrill/test_cor'
input_shape = (img_height, img_width, 1)


# NOTE: EITHER RESCALE ALL IMAGES TO FIXED SIZE
# OR GET THE MAX WIDTH AND HEIGHT OF IMAGES
# EITHER RESIZE OR CROP/PAD
def load_data(train=True):
    # if train:
    target_path = train_dir
    # else:
    #     target_path = test_dir
    x, y = image_preloader(target_path=target_path,
                           image_shape=(img_height, img_width),
                           mode='folder',
                           categorical_labels=True,
                           normalize=True,
                           grayscale=True)
    
    x = np.asarray(x[:], dtype='float32')
    y = np.asarray(y[:], dtype='float32')

    return x, y


x, y = load_data()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
x_train = x_train.reshape(x_train.shape[0], img_height, img_width, 1)
x_test = x_test.reshape(x_test.shape[0], img_height, img_width, 1)

# x_test = x_test.reshape(x_test.shape[0], img_height, img_width, 1)

# print(type(y_train))
# class_weights = class_weight.compute_class_weight('balanced',
#                                                  np.unique(y_train),
#                                                  y_train)
# class_weights = class_weight.compute_sample_weight(class_weight='balanced',
#                                                    y = y_train)
# print(len(class_weights))
# print(class_weights[:30])


# Initialize model
def init_model():
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

    return model


model = lenet5()
print(model.summary())
# model.compile(loss=keras.losses.categorical_crossentropy,
#                   optimizer=keras.optimizers.Adam(),
#                   metrics=['accuracy'])

# checkpoint
filepath="lenet5.equalclasses.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Data augmentation
gen = ImageDataGenerator(rotation_range=8,
                         width_shift_range=0.08,
                         shear_range=0.3,
                         height_shift_range=0.08,
                         zoom_range=0.08)
test_gen = ImageDataGenerator()

train_generator = gen.flow(x_train, y_train, batch_size=batch_size)
test_generator = test_gen.flow(x_test, y_test, batch_size=batch_size)

# class_weights = max(np.sum(y_train, axis=0)) / np.sum(y_train, axis=0) 
# print(np.sum(y_train, axis=0))
# print(class_weights)

# Run model
history = model.fit_generator(train_generator,
                              steps_per_epoch=x_train.shape[0] // batch_size,
                              epochs=num_epochs,
                              validation_data=test_generator,
                              validation_steps=x_test.shape[0] // batch_size,
                              callbacks=callbacks_list)


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# # tmp = x_train[0].reshape(48, 32, 1, 1)
# # prediction = model.predict(np.asarray([x_train[0]]))
# # for i in range(len(prediction)):
# # 	print('Predicted: ', prediction[i])


# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()