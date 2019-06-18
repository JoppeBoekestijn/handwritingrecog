from scipy.signal import find_peaks

# Sums all the black pixels per row in a binary image
def getHist(inputimg, mask):    

    #inputimg = cv2.bitwise_not(inputimg)
    #kernel2 = np.array([[1,1,1],[1,1,1],[1,1,1]])
    #inputimg = cv2.dilate(inputimg,kernel2,iterations = 2)
    
    if np.amax(inputimg) == 255:
        binary_img = inputimg / 255
    else:
        binary_img = inputimg  
        
    height, width = binary_img.shape
    
    hist = []
    for x in range(height):
        # find indexes of start and end of paper
        pixels = np.where(mask[x] == 255)
        
        if len(pixels[0])>0:
            rightp = pixels[0][-1]
            leftp = pixels[0][0]
            #calculate paper width
            paperw = rightp-leftp
            
            if paperw > 10:
                #sum number of white pixels
                rowsum = binary_img[x].sum()
                normsum = rowsum/paperw
                
                if normsum > 1:
                    hist.append(hist[-1])
                    print(normsum)
                else:
                    hist.append(normsum)
                    
            else:
                hist.append(0)
                
        else:
            hist.append(0)
            
    return hist



# Finds the coordinates of where to slice
def findSlices(histogram):
	peaks, _ = find_peaks(histogram, width=5, distance = len(hist)/40, height = max(hist)/5)
	slices = []
	
	#find start coordinate of paper
	for x in range(len(histogram)):
		if histogram[x] > 0:
			slices.append(x)
			break
	
	#find slicing locations
	for peak in range(len(peaks)-1):
		slice = int(peaks[peak] + peaks[peak+1] / 2)
		slices.append(slice)
	
	#find end coordinate of paper
	for x in reversed(range(len(histogram))):
		if histogram[x] > 0:
			slices.append(x)
			break
	
	return slices

#slices papyrus into little papyri
#def sliceImg(inputimg, slices, folder, original, mask):
def sliceImg(inputimg, original, mask, slices):
	slicesList = []
	if np.amax(inputimg) == 1:
		paper = inputimg * 255
	else:
		paper = inputimg
	height, width = paper.shape

    #paper = cv2.bitwise_not(paper)
	mask = mask.astype('uint8')
	original = cv2.bitwise_and(original, original, mask=mask)
	kleuren = np.unique(original)
    #whiteor[:] = median(kleuren[int(len(kleuren)/2):])
	whiteor = np.random.randint(kleuren[int(len(kleuren)/2)],kleuren[-1], size=(height, width,3),dtype=np.uint8)
    
	whiteor = cv2.bitwise_and(whiteor,whiteor, mask=cv2.bitwise_not(mask))
	original = cv2.add(whiteor,original)
    
	for slice in range(len(slices)-1):
		upper = slices[slice]
		lower = slices[slice+1]
		slicesList.append(original[upper:lower])
        #check if slice is big enough
        #if slices[y]-lastslice > height/(len(slices)*4):
        #cv2.imwrite(folder+'slice'+str(y)+'.png',paper[lastslice:slices[y]])
        #cv2.imwrite(folder+'slice'+str(y)+'.png',original[lastslice:slices[y]])        
        #binarizeSlice(folder+'slice'+str(y)+'.png', folder, y)
      #  binarizeSliceOtsu(folder+'slice'+str(y)+'.png', folder, y)
	
	return slicesList



