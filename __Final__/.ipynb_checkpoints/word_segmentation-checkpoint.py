# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.3
#   kernel_info:
#     name: python3
#   kernelspec:
#     display_name: Python [conda env:tensorflow]
#     language: python
#     name: conda-env-tensorflow-py
# ---

# + {"outputHidden": false, "inputHidden": false}
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

print(cv2.__version__)


# + {"outputHidden": false, "inputHidden": false}
def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = io.BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))


# + {"outputHidden": false, "inputHidden": false}
model = load_model('../weights.best.hdf5')
# new_model = tf.keras.experimental.load_from_saved_model(saved_model_path)
model.summary()


# + {"outputHidden": false, "inputHidden": false}


# + {"outputHidden": false, "inputHidden": false}
img = cv2.imread('input_files_word_segment/slice5.png')  #Afbeelding waar je alles op uitvoert

# convert to grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# smooth the image to avoid noises
gray = cv2.medianBlur(gray,5)

# Apply adaptive threshold
thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
thresh_color = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)

# apply some dilation and erosion to join the gaps
thresh = cv2.dilate(thresh,None,iterations = 3)
thresh = cv2.erode(thresh,None,iterations = 2)

# Find the contours
_, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

thresh = img.copy()

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# For each contour, find the bounding rectangle and draw it

# + {"outputHidden": false, "inputHidden": false}
# ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# thresh1.shape
showarray(img)

# + {"outputHidden": false, "inputHidden": false}
words = [];
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    if w > 50:
        words.append(img[y:y+h, x:x+w]);
    cv2.rectangle(thresh,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.rectangle(thresh_color,(x,y),(x+w,y+h),(0,255,0),2)


# + {"outputHidden": false, "inputHidden": false}
word = words[2]
showarray(word)

# + {"outputHidden": false, "inputHidden": false}
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

# + {"outputHidden": false, "inputHidden": false}
h, w, _ = word.shape
num = 2
chars = []
for i in range(num):
    part = math.floor(w / num)
    char = word[:,part * i:(part * i) + part]
    shape = cv2.resize(char,(32,48))
    ret,thresh1 = cv2.threshold(shape,127,255,cv2.THRESH_BINARY)
    chars.append(thresh1)

# + {"outputHidden": false, "inputHidden": false}
char = chars[1]
showarray(char)

# + {"outputHidden": false, "inputHidden": false}
char = cv2.cvtColor(char, cv2.COLOR_BGR2GRAY)
char = np.asarray(char[:], dtype='float32')
char = char.reshape((-1, 48, 32,1))

# + {"outputHidden": false, "inputHidden": false}
with tf.device('/cpu:0'):
      model.predict(char)
model.predict([char]).shape
print(model.predict([char]))

# + {"outputHidden": false, "inputHidden": false}
# for box in bounding_boxes:
#     xStart = box[2]
#     xEnd = box[0]
#     y = box[1]
#     winH = box[3] - y
#     winWidth = 5
#     while(xStart-winWidth >= xEnd) :
#         hit = False
#         winW = winWidth
#         a = 0
#         # While the image is not classified and the box has not reached the edge,
#         # increase window size
#         while(not hit and xStart-winW >= xEnd) :
#             newX = xStart - winW
#             # Draw the window
#             clone = img.copy()
#             cv2.rectangle(clone, (xStart, y), (newX, y + winH), (255, 0, 0), 2)
#             cv2.rectangle(clone, (xStart,y),(xEnd,y + winH), (0,255,0), 2)
#             cv2.imshow("Window", clone)
#             cv2.waitKey(0)
#             # Check if the CNN returns a high probability for a letter
#             # for prob in probabilities :
#             #     if prob >= 0.75 :
#             #         hit = True
#             #         xStart = newX
#             # # Increase size of window if nothing has been found
#             winW += 5
#             # this is done to ensure that the loop ends for now, because not
#             # connected to cnn yet.
#             hit = True
#             xStart = newX

# + {"outputHidden": false, "inputHidden": false}


# + {"outputHidden": false, "inputHidden": false}


# + {"outputHidden": false, "inputHidden": false}


# + {"outputHidden": false, "inputHidden": false}

