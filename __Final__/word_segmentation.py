# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.6
#   kernel_info:
#     name: python3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + {"inputHidden": false, "outputHidden": false}
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.data import page
from skimage.filters import threshold_sauvola

import io
from IPython.display import clear_output, Image, display
import PIL.Image
from keras.models import load_model
import math
import tensorflow as tf
from skimage.color import rgb2gray
# -

from util.WordSegmentation import wordSegmentation, prepareImg

print(cv2.__version__)


# + {"inputHidden": false, "outputHidden": false}
def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = io.BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))


# + {"inputHidden": false, "outputHidden": false}
model = load_model('../weights.best.hdf5')
# new_model = tf.keras.experimental.load_from_saved_model(saved_model_path)
model.summary()


# + {"inputHidden": false, "outputHidden": false}
import_images = []
import_images.append(prepareImg(cv2.imread('input_files_word_old/slice4.png'), 50))
# -

res = []
for i, img in enumerate(import_images):
    res = wordSegmentation(img, kernelSize=5, sigma=5, theta=7, minArea=30) # fix parameters

for (j, w) in enumerate(res):
    (wordBox, wordImg) = w
    (x, y, w, h) = wordBox
    if (w > 20):
        words.append(wordImg)
        

# + {"inputHidden": false, "outputHidden": false}
word = words[1]
showarray(word)

# + {"inputHidden": false, "outputHidden": false}
# def get_resized_img(img, video_size):
#     width, height = video_size  # these are the MAX dimensions
#     video_ratio = width / height
#     img_ratio = img.size[0] / img.size[1]
#     if video_ratio >= 1:  # the video is wide
#         if img_ratio <= video_ratio:  # image is not wide enough
#             width_new = int(height * img_ratio)
#             size_new = width_new, height
#         else:  # image is wider than video
#             height_new = int(width / img_ratio)
#             size_new = width, height_new
#     else:  # the video is tall
#         if img_ratio >= video_ratio:  # image is not tall enough
#             height_new = int(width / img_ratio)
#             size_new = width, height_new
#         else:  # image is taller than video
#             width_new = int(height * img_ratio)
#             size_new = width_new, height
#     return np.asarray(img.resize(size_new, resample=Image.LANCZOS))

# + {"inputHidden": false, "outputHidden": false}
h, w, _ = word.shape
num = 2
chars = []
for i in range(num):
    part = math.floor(w / num)
    char = word[:,part * i:(part * i) + part]
    shape = cv2.resize(char,(32,48))
    ret,thresh1 = cv2.threshold(shape,127,255,cv2.THRESH_BINARY)
    chars.append(thresh1)

# + {"inputHidden": false, "outputHidden": false}
char = chars[1]
showarray(char)

# + {"inputHidden": false, "outputHidden": false}
char = cv2.cvtColor(char, cv2.COLOR_BGR2GRAY)
char = np.asarray(char[:], dtype='float32')
char = char.reshape((-1, 48, 32,1))

# + {"inputHidden": false, "outputHidden": false}
with tf.device('/cpu:0'):
      model.predict(char)
model.predict([char]).shape
print(model.predict([char]))

# + {"inputHidden": false, "outputHidden": false}
for box in bounding_boxes:
    xStart = box[2]
    xEnd = box[0]
    y = box[1]
    winH = box[3] - y
    winWidth = 5
    while(xStart-winWidth >= xEnd) :
        hit = False
        winW = winWidth
        a = 0
        # While the image is not classified and the box has not reached the edge,
        # increase window size
        while(not hit and xStart-winW >= xEnd) :
            newX = xStart - winW
            # Draw the window
            clone = img.copy()
            cv2.rectangle(clone, (xStart, y), (newX, y + winH), (255, 0, 0), 2)
            cv2.rectangle(clone, (xStart,y),(xEnd,y + winH), (0,255,0), 2)
            cv2.imshow("Window", clone)
            cv2.waitKey(0)
            # Check if the CNN returns a high probability for a letter
            # for prob in probabilities :
            #     if prob >= 0.75 :
            #         hit = True
            #         xStart = newX
            # # Increase size of window if nothing has been found
            winW += 5
            # this is done to ensure that the loop ends for now, because not
            # connected to cnn yet.
            hit = True
            xStart = newX

# + {"inputHidden": false, "outputHidden": false}


# + {"inputHidden": false, "outputHidden": false}


# + {"inputHidden": false, "outputHidden": false}


# + {"inputHidden": false, "outputHidden": false}

