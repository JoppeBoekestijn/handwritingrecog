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
import io
import PIL.Image
import math
import glob
import tensorflow as tf

from matplotlib import pyplot as plt
from IPython.display import clear_output, Image, display
from keras.models import load_model
from util.WordSegmentation import wordSegmentation, prepareImg
from sklearn.preprocessing import normalize
from natsort import natsorted
from keras.backend.tensorflow_backend import set_session
# -

config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

class_names = [
    "Alef",
    "Ayin",
    "Bet",
    "Dalet",
    "Gimel",
    "He",
    "Het",
    "Kaf",
    "Kaf-final",
    "Lamed",
    "Mem",
    "Mem-medial",
    "Nun-final",
    "Nun-medial",
    "Pe",
    "Pe-final",
    "Qof",
    "Resh",
    "Samekh",
    "Shin",
    "Taw",
    "Tet",
    "Tsadi-final",
    "Tsadi-medial",
    "Waw",
    "Yod",
    "Zayin"
]


# + {"inputHidden": false, "outputHidden": false}
def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = io.BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))

# + {"inputHidden": false, "outputHidden": false}
# model = load_model('temporary.best.hdf5')
# new_model = tf.keras.experimental.load_from_saved_model(saved_model_path)
# model.summary()


# + {"inputHidden": false, "outputHidden": false}
# import_images = []
# import_images.append(prepareImg(cv2.imread('input_files_word_old/slice4.png'), 50))
# showarray(import_images[0])

# +
input_files = 'output_files/image-data/'

def load_slices(filepath):
    slices = []
    slices_filepath = filepath + '/'
    slices_path = natsorted(glob.glob(slices_filepath + 'slice*_binarize.png'))
    for slice_path in slices_path:
        slices.append(prepareImg(cv2.imread(slice_path), 50))  
    #print(len(slices))
    return slices

image_filepaths = glob.glob(input_files + '*')
print(image_filepaths)
print(image_filepaths[len(image_filepaths) - 5])
images = []
for image_filepath in image_filepaths:
    images.append(load_slices(image_filepath))

# -

#Count the number of black pixels in an image and return a float with the density
def pixel_density(image):
    area = float(image.shape[0]*image.shape[1])
    blackPixels = float(np.sum(image == 0))
    density = blackPixels/area
    return density


# +
# """
# res = []
# words = []
# for i, img in enumerate(import_images):
#     res = wordSegmentation(img, kernelSize=5, sigma=5, theta=7, minArea=30) # fix parameters
# """

# +
#There still needs to come a loop through all the slices, here you can pick a slice to debug
# img = images[12][5]
# showarray(img)
#Parameters are optimized. 
segments = []
for (i, img) in enumerate(images):
    for (j, slic) in enumerate(img):
        res = wordSegmentation(slic, kernelSize=13, sigma=3, theta=((slic.shape[1] / slic.shape[0]) / 4), minArea=15 * 15)
        segments.append({
            "res": res,
            "img": i,
            "slic": j
        })
        


# +
segments_1 = []
for (i, seg) in enumerate(segments):
    words = []
    places = []
    for (j, w) in enumerate(seg["res"]):
        (wordBox, wordImg) = w
        (x, y, w, h) = wordBox
        words.append(wordImg)
        segments_1.append({
            "img": seg["img"],
            "slic": seg["slic"],
            "word": wordImg,
            "word_place": y
        })
        
        


# + {"inputHidden": false, "outputHidden": false}
segments_2 = []

for (i, seg) in enumerate(segments_1):
    res = wordSegmentation(seg["word"], kernelSize=5, sigma=3, theta=1, minArea=70) # fix parameters

    for (j, w) in enumerate(res):
        (charBox, charImg) = w
        (x, y, w, h) = charBox

        if( pixel_density(charImg) < 0.55):
            segments_2.append({
                "img": seg["img"],
                "slic": seg["slic"],
                "word_place": seg["word_place"],
                "word": seg["word"],
                "char_place": y,
                "char": charImg
            })
            

# +
#Look for the bounding boxes over the words again to possible find characters. It will return the box of the entire word
#when it does not find seperate characters in the word. 
#The character needs a minimum width of 15, height of 10 and maximum density of 55%


segments_3 = []
average_char_width = 17
offset_rounding_err = 0.3

for (i, seg) in enumerate(segments_2):
    char = seg["char"]
    factor = round(((char.shape[1]) / average_char_width) - offset_rounding_err)
    if factor > 1:
        for i in range(factor):
            full_width = char.shape[1]
            width = math.floor(full_width / factor)
            slic = width * i;
            new_char = char[:, slic:slic+width]
            
            segments_3.append({
                "img": seg["img"],
                "slic": seg["slic"],
                "word_place": seg["word_place"],
                "word": seg["word"],
                "char_place": seg["char_place"] + slic,
                "char": new_char
            })
    else:
        segments_3.append(seg);
        
# -

for seg in segments_3:
 showarray(seg["char"])

# + {"inputHidden": false, "outputHidden": false, "cell_type": "markdown"}
# ##### word = words[2]
# h, w = word.shape
# num = 4
# chars = []
# for i in range(num):
#     part = math.floor(w / num)
#     char = word[:,part * i:(part * i) + part]
#     shape = cv2.resize(char,(32,48))
#     ret,thresh1 = cv2.threshold(shape,127,255,cv2.THRESH_BINARY)
#     chars.append(thresh1)

# + {"inputHidden": false, "outputHidden": false}


# +
# import os
# import ipywidgets as widgets
# import random
# import uuid

# if not os.path.isdir("generated"):
#     os.mkdir("generated")
# hebrews = os.listdir("habbakuk")
# chosen = ""

# def on_button_clicked(v):
#     if not os.path.isdir("generated/" + v.description):
#         os.mkdir("generated/" + v.description)
        
#     cv2.imwrite("generated/" + v.description + "/" + str(uuid.uuid1()) + ".png", seg["char"]);
#     clear_output()
#     generate(i)
    
# def undo_button_clicked(d):
#     clear_output()
#     generate(i)
    
# def generate(index):
#     global chosen
#     seg = random.choice(segments_3)
#     chosen = seg
#     showarray(seg["word"])
#     showarray(seg["char"])
#     undo = widgets.Button(
#         description="I don't know"    
#     )
#     undo.on_click(undo_button_clicked)
#     display(undo)
#     wids = []
#     for (i, hebrew) in enumerate(hebrews):
#         file = open("habbakuk/" + hebrew + "/img_modefilter_5.png", "rb")
#         image = file.read()
#         img = widgets.Image(
#             value=image,
#             format='png',
#             width=40,
#             height=440,
#         )
#         btn = widgets.Button(
#             description=hebrew,
#         )
#         btn.on_click(on_button_clicked)
#         wids.append(widgets.HBox([img, btn]))
        
#         if i % 5 == 4:
#             display(widgets.HBox(wids))
#             wids = []
#     display(widgets.HBox(wids))
            
# generate(i)
    



# + {"inputHidden": false, "outputHidden": false}
fonts = []
for item in class_names:
    fonts.append(cv2.imread('habbakuk/' + item + '/standard.png'))

# + {"inputHidden": false, "outputHidden": false}
for seg in segments_3:
    char = seg["char"]
#     char_pred = cv2.cvtColor(char, cv2.COLOR_BGR2GRAY)
    char_pred = np.asarray(char[:], dtype='float32')
    print(char_pred.shape)
    char_pred = normalize(char_pred)
    char_pred = char_pred.reshape(-1, 48, 32,1)

    prediction = model.predict([char_pred])
    for i in range(len(prediction)):
        print('Predicted: ', prediction[i] * 100)
    highest_index = np.argmax(prediction)
    print('Index of class with highest probability: ',highest_index)
    print('Value of highest probability: ', prediction[0][highest_index])
    print('Name of predicted class: ', class_names[highest_index])
    print('habbabuk/' + class_names[highest_index] + '/standard.png')
    character_example = cv2.imread('habbakuk/' + str(class_names[highest_index]) + '/standard.png')
    showarray(character_example)
    showarray(char)

# + {"inputHidden": false, "outputHidden": false}
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
# char = cv2.cvtColor(char, char, cv2.COLOR_BGR2GRAY)


# + {"inputHidden": false, "outputHidden": false}



# + {"inputHidden": false, "outputHidden": false}



# + {"inputHidden": false, "outputHidden": false}

# -


















