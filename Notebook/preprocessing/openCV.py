import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.data import page
from skimage.filters import threshold_sauvola

#Gemaakt met
#Python Version: 2.7.12
#OpenCV Version: 3.4.4

def undesired_objects (image):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[:, -1]
    #Show components of the image
    imshow_components(output)

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
            
    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255
    cv2.imwrite("output_files/biggest_component.png", img2)
    return img2

def hole_removal (image, percentile):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[:, -1]
    #Show components of the image
    imshow_components(output)
    #Calculate the threshold to remove holes
    sizes_copy = np.delete(sizes, np.argmax(sizes))
    avg = np.average(sizes_copy)
    #The x percentile is used as a threshold
    val = np.percentile(sizes_copy, percentile)
    var_threshold = val
    print(avg)
    print(val)
    #We also create an absolute threshold to not accidentally remove small characters
    abs_threshold = 1500
    #Calculate the pixel density for each component
    pixel_density = np.zeros(nb_components+1)
    for i in range(2, nb_components):
        #convert connected component to an image and get white pixel density
        img1 = np.zeros(output.shape)
        img1[output == i] = 255
        cv2.imwrite("output_files/biggest_componenttemp.png", img1)
        pixel_density[i] = pixel_density_connected_component(img1, image)
        #raw_input("Press Enter to continue...")
    
    #Calculate the desnity threshold with a percentile. 
    density_threshold = np.percentile(pixel_density, 25)
    #Create empty image, only add components smaller than the threshold
    img2 = np.zeros(output.shape)
    for i in range(2, nb_components):
        if sizes[i] <= var_threshold:
            img2[output == i] = 255
        elif sizes[i] <= abs_threshold:
            img2[output == i] = 255
        #elif sizes[i] > var_threshold:
         #   if pixel_density[i] <= density_threshold:
         #       img2[output == i] = 255
    return img2

def pixel_density_connected_component (connectedComponent, image):
	#Make sure image is of correct type
	connectedComponent = connectedComponent.astype('uint8')
	
	#Create bounding box around largest component with contours
	_, contours, hierarchy = cv2.findContours(connectedComponent, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

	#Get the appropriate coordinates
	x = bounding_boxes[0][0]
	y = bounding_boxes[0][1]
	width = bounding_boxes[0][2]
	height = bounding_boxes[0][3]
	
	#Crop image to bounding box arround component
	crop_img = image[y:y+height, x:x+width]
	cv2.imwrite('output_files/cropped_component.png', crop_img)
	#Calculate density of the this newly create image of component
	area = float(crop_img.shape[0]*crop_img.shape[1])
	whitePixels = float(np.sum(crop_img == 255))
	density = whitePixels/area
	#print("whitePixels: {}".format(whitePixels))
	#print("area: {}".format(area))
	#print("density: {}".format(density))
	
	return density

def inverted(imagem):
    imagem = (255-imagem)
    return imagem
    
def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    cv2.imwrite('output_files/connected_components.png', labeled_img)

def img_fill(im_th):  # n = binary image threshold
    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    fill_image = im_th | im_floodfill_inv

    return fill_image 

#Read image, and perform binarization using Otsu algorithm
img = cv2.imread('input_files/test9.jpg')  #Afbeelding waar je alles op uitvoert
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Not using these two lines yet.
ret, thresholded_img = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
cv2.imwrite('output_files/thresholded_img.png',thresholded_img)

#Perform OTSU binarization
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#Show  and save image
cv2.imwrite('output_files/otsu.png',thresh)
#Find connected components of image
output_components = cv2.connectedComponentsWithStats(cv2.bitwise_not(thresh))
# The first cell is the number of labels
num_labels = output_components[0]
# The second cell is the label matrix
labels = output_components[1]
# The third cell is the stat matrix
stats = output_components[2]
# The fourth cell is the centroid matrix
centroids = output_components[3]

#Invert the binarized image in order to be able to use connected components
inverted = cv2.bitwise_not(thresh)
cv2.imwrite('output_files/inverted_otsu.png', inverted)

#Get the biggest components, the dead sea scroll.
largest_component = undesired_objects(inverted)
#Convert to right type
largest_component = largest_component.astype('uint8')
#Save file for debugging purposes. 
cv2.imwrite('output_files/is_dit_het_nou.png', largest_component)

#Create bounding box around largest component with contours
largest_component_copy = largest_component
_, contours, hierarchy = cv2.findContours(largest_component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

#I was thinking to use contours to cut out the piece of scroll rather than just
#taking the largest component. But have yet to implement this. 
img_temp = np.zeros(img.shape)
img_contour_tmp = cv2.drawContours(img_temp, contours, -1, [255,255,255], -1)
cv2.imwrite('output_files/countour_tmp.png', img_contour_tmp)

#Get the appropriate coordinates
x = bounding_boxes[0][0]
y = bounding_boxes[0][1]
width = bounding_boxes[0][2]
height = bounding_boxes[0][3]

#Crop from the originalimage to only get the scroll wihtou losing information at the edges
#crop_img = inverted[y:y+height, x:x+width]

#We decide to not crop to image to have the image be the same size through the entire pipeline.
#So we just take the largest component to further process.
crop_img = largest_component
cv2.imwrite('output_files/crop.png', crop_img)

#Draw rectangle around the largest component to see what is being cropped. 
#cv2.rectangle(largest_component_copy,(x,y),(x+width,y+height),(255,0,0),2)
#cv2.imwrite('output_files/vierkant.png', largest_component_copy)
#crop_img2 = img[y:y+height, x:x+width]

###################################################################
kernel2 = np.array([[1,1,1],[1,1,1],[1,1,1]])
kernel3 = np.array([[1,1],[1,1]])
dilated_img = cv2.dilate(crop_img,kernel2,iterations = 2)
cv2.imwrite('output_files/dilated.png', dilated_img)

#Get the countours of the scroll
_, contours, hierarchy = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
img2 = np.zeros(crop_img.shape)
img_contour = cv2.drawContours(img2, contours, -1, 255, -1)

#dilate and erode to fill up the edges of the scroll.
img_contour = cv2.dilate(img_contour,kernel2,iterations = 6)
img_contour = cv2.erode(img_contour,kernel2,iterations = 10)
cv2.imwrite('output_files/contours.png', img_contour)

#Convert the shape of the scroll to binarized image and invert.  
ret, mask = cv2.threshold(img_contour, 127, 255, cv2.THRESH_BINARY)
mask = mask.astype('uint8')
mask = cv2.bitwise_not(mask)
cv2.imwrite('output_files/mask.png', mask)

#Add the mask to the original image to delete background. 
added = crop_img + mask
#added = cv2.morphologyEx(added, cv2.MORPH_CLOSE, kernel3, iterations = 3)
#added = cv2.morphologyEx(added, cv2.MORPH_CLOSE, kernel2, iterations = 1)
cv2.imwrite('output_files/added.png', added)

#Get rid of noise, especially at the edges
added = cv2.morphologyEx(added, cv2.MORPH_CLOSE, kernel2, iterations = 1)

#To debug: show largest connected component of the result. 
mask = undesired_objects(cv2.bitwise_not(added))
cv2.imwrite('output_files/added_inv.png', cv2.bitwise_not(added))

output_hole_removal = hole_removal(cv2.bitwise_not(added), 99.4)
output_hole_removal = output_hole_removal.astype('uint8')
cv2.imwrite("output_files/holes_removed.png", cv2.bitwise_not(output_hole_removal))
 
###################################################################

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(crop_img,cv2.MORPH_OPEN,kernel, iterations = 1)
#Save image for debugging
cv2.imwrite('output_files/opening.png',opening)
# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=1)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
#Save image for debugging
cv2.imwrite('output_files/dist_transform.png',dist_transform)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
#Save images to visualize
cv2.imwrite('output_files/fg.png',sure_fg)
cv2.imwrite('output_files/bg.png',sure_bg)
cv2.imwrite('output_files/unkown.png',unknown)


# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0
cv2.imwrite('output_files/markers.png',markers)

#Perform watershed and save the result. 
markers = cv2.watershed(img,markers)
img[markers == -1] = [0,255,0]
cv2.imwrite('output_files/watershed_output.png',img)

