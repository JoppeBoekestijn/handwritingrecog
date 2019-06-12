import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
from tflearn.data_utils import image_preloader

num_classes = 27
img_width = 32
img_height = 48
train_dir = 'figures_monkbrill/train_cor'
test_dir = 'figures_monkbrill/test_cor'
input_shape = (img_height, img_width, 1)


def load_data(train=True):
    if train:
        target_path = train_dir
    else:
        target_path = test_dir
    x, y = image_preloader(target_path=target_path,
                           image_shape=(img_height, img_width),
                           mode='folder',
                           categorical_labels=True,
                           normalize=True,
                           grayscale=True)

    x = np.asarray(x[:], dtype='float32')
    y = np.asarray(y[:], dtype='float32')
    return x, y


def main():
    x_test, y_test = load_data(train=False)
    x_test = x_test.reshape(x_test.shape[0], img_height, img_width, 1)
    print(x_test.shape[0])

    model = load_model('temporary.best.hdf5')

    # score = model.evaluate(x_test, y_test, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])

    test_image_number = 120
    prediction = model.predict(np.asarray([x_test[test_image_number]]))
    prediction_class = model.predict_classes(np.asarray([x_test[test_image_number]]))
    print('Predicted class', prediction_class)
    
    for i in range(len(prediction)):
        print('Predicted: ', prediction[i])
    print(np.where(y_test[test_image_number] == 1)) 
    print(sum(prediction))


if __name__ == '__main__':
    main()
