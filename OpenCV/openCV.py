import numpy as np
import cv2
from matplotlib import pyplot as plt

#Read image, and perform binarization using Otsu algorithm
img = cv2.imread('klein.jpg')  #Afbeelding waar je alles op uitvoert
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#Show  and save image
#cv2.imshow('image',thresh)
#cv2.waitKey(0)
cv2.imwrite('otsu.png',thresh)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
#Save image for debugging
cv2.imwrite('test1.png',opening)
# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
#Save image for debugging
cv2.imwrite('test3.png',dist_transform)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

cv2.imwrite('fg.png',sure_fg)
cv2.imwrite('bg.png',sure_bg)
cv2.imwrite('unkown.png',unknown)


# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0
cv2.imwrite('markers.png',markers)

#Perform watershed and save the result. 
markers = cv2.watershed(img,markers)
img[markers == -1] = [0,255,0]
cv2.imwrite('img.png',img)
