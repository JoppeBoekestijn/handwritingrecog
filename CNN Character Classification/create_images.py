import glob
import random
import cv2
import os 
import numpy as np
from skimage import transform
from skimage import util
from scipy.ndimage.interpolation import zoom, map_coordinates
from scipy.ndimage.filters import gaussian_filter
from PIL import Image

input_filepath = 'figures_monkbrill/train_cor/'
character_filepaths = glob.glob(input_filepath + '*')


def random_rotation(img):
	img = np.invert(img)
	angle = random.uniform(-10, 10)
	(height, width) = img.shape[:2]
	(cent_x, cent_y) = (width // 2, height // 2)

	mat = cv2.getRotationMatrix2D((cent_x, cent_y), -angle, 1.0)
	cos = np.abs(mat[0, 0])
	sin = np.abs(mat[0, 1])

	n_width = int((height * sin) + (width * cos))
	n_height = int((height * cos) + (width * sin))

	mat[0, 2] += (n_width / 2) - cent_x
	mat[1, 2] += (n_height / 2) - cent_y

	img = cv2.warpAffine(img, mat, (n_width, n_height))
	return np.invert(img)


def random_noise(image):
    return util.random_noise(image)


def shift_img(image, hor=True):
	rows, cols, channels = image.shape
	image = np.invert(image) # Inverted such that correct color pixels are added

	random_shift = random.uniform(-5, 5)
	if hor:
		M = np.float32([[1, 0, 0],[0, 1, random_shift]])
	else: # Shift must be vertical
		M = np.float32([[1, 0, random_shift],[0, 1, 0]])

	image = cv2.warpAffine(image, M, (cols, rows))
	return np.invert(image) # Invert back to black-on-white


for character_filepath in character_filepaths:
	image_filepaths = glob.glob(character_filepath + '/*')
	image_names = [os.path.basename(x) for x in image_filepaths]
	number_images = len(image_filepaths)
	print(image_filepaths)

	while number_images < 300: # number 300 must be changed later
		for idx, image_filepath in enumerate(image_filepaths):
			path = os.path.splitext(image_filepath)[0]
			print('path: ', path)
			if number_images < 300:
				# try: 
				image = cv2.imread(image_filepath)
				# tmp = Image.fromarray(random_rotation(image))
				# tmp.save('tmp.pgm')
				tmp = Image.fromarray(random_rotation(image))
				tmp.save(path + '_rot' + '.pgm')
				tmp = Image.fromarray(shift_img(image, hor=False))
				tmp.save(path + '_shiftvertdown' + '.pgm')
				Image.fromarray(shift_img(image, hor=False)).save(path + '_shiftvertup' + '.pgm')
				Image.fromarray(shift_img(image)).save(path + '_shifthordown' + '.pgm')
				Image.fromarray(shift_img(image)).save(path + '_shifthorup' + '.pgm')
				list = os.listdir(character_filepath) 
				number_images = len(list)
				# except:
				# 	print(idx, image_filepath)

	print(len(image_filepaths))


# [300. 300. 300.  91. 300. 300. 300. 193.  10. 294. 300. 300. 130. 300.
#   37.  15. 265.  78. 300. 300. 300. 300.  73. 300. 116.  23.  12.]

# cv2.imshow('image', random_rotation(image))
				# cv2.waitKey(0)
				# cv2.destroyAllWindows()
				# cv2.imshow('image', random_noise(image))
				# cv2.waitKey(0)
				# cv2.destroyAllWindows()
				# cv2.imshow('image', shift_img(image, hor=False))
				# cv2.waitKey(0)
				# cv2.destroyAllWindows()
				# cv2.imshow('image', shift_img(image, hor=False))
				# cv2.waitKey(0)
				# cv2.destroyAllWindows()
				# cv2.imshow('image', shift_img(image))
				# cv2.waitKey(0)
				# cv2.destroyAllWindows()
				# cv2.imshow('image', shift_img(image))
				# cv2.waitKey(0)
				# cv2.destroyAllWindows()