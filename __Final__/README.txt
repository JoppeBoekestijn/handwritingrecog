# How to run the code? 


# Main file
We start with asking for the folder where the input images are located. Then we check that folder and see which files are inside. We try to find all jpg images. Nestor stated there should be 20 images, so we check for that.
We then use this list to load each image into our recognition system and let it process the image. After the pre-processing, line segmentation, classification and the language model we output the classified characters to a text file and save that under the same name as the input image. 


# Pre processing file
The preprocessing starts with the images provided in black and white and binarizes those in 2 different manners. First we isolate the papyrus through taking the largest connectected component and the background will be completely removed. A number of masks are created acocrdingly, for example one for only the papyrus but also 2 masks with the results of two different methods of hole removal. These two masks will be combined and the result is 1 mask representing all the holes in the papyrus as far as the code can detect. All the masks and images are optimized and cleaned using morphological transformations such as dilation and erosion. The hole mask will be used in the line segmentation stage seeing as we will binarize the images again there per slice and we need to remove the holes again. 
The first method of hole removal is to look at all the connected components in the image seeing as all we are left with at this stage is a white image with text and holes in black. Then the components in the 99.9th percentile are removes seeing as this will be noise. The 99.9% percentile is rather conversvative but we dicided that having some noise here and there is better than accidentally removing characters. For the component to be removed it does have to be larger than an absolute threshold as well.
For the second hole removal method a second method of binarizing the image is used with a manually set threshold rather than the adaptive threshold (OTSU) used in the first method. This result is combined with a mask representing the papyrus and this will filter out holes regardless of their size but rather based on their color. 
Ultimately the resulting cleaned image will be used in order to determine where the lines should be segmented. And the masks created will be used after the line segmentation stage. 

# Line segmentation file
The line segmentation receives the binarized image and a mask from the pre-processing stage. 
The first step in the line segmentation is to transform the image to a feature space where we can derive information from about where in the image the lines are. This is achieved by summing all ink pixels per horizontal row of one pixel height. What we then get is a histogram-like graph with pixel densities. This would provide us with decent results on a regular, rectangular a4 paper. But, we are working with fragments of the dead sea scrolls. These are very irregular shaped pieces of paper with differing widths. Therefore we add a normalization to the summation. This is done by dividing the number of found ink pixels through the paper width at that position.
We then try to find the most prominent peaks by examining the generated histogram. This is done using a peak finding algorithm with three parameters: width, distance and height. Width resembles the width of the peak and disregards every peak with a width lower than 5. Distance is the minimum distance the peaks should appear from each other. This is computed by dividing the height of the image through the maximum number of lines that appeared in the provided dataset plus a small buffer. The height parameter states the minimum height that the normalized peak should have. We found that it should not be smaller than 1/5th of the highest peak. 
The coordinates of the most prominent peaks are used to determine the slicing coordinates. These are calculated by finding the exact middle distance between two peaks. So two peak coordinates are added and divided by two. This method was created under the, rather strong, assumption that each new line in the image is spaced to a constant ratio. 
These three steps leaves us with a number of slices ready to be processed in the next step.

# Binarization file
In the line segmentation it is determined where the image has to be sliced in order to capture all the lines separately. Then rather than using the cleaned image we cut up the orignal image and for each seperate slice we binarize it again. The holes will be removed using the mask created in the preprocessing stage. Then each slice is cleaned of noise as properly as possible and these slices will serve as input for the word and character segmentation stages. 

# Word and character segmentation


# classification file


# Language model file
The word classified with the neural network is sent to the language model. This model first checks if the word occurs in the provided ngram dictionary. If it is mentioned in the dictionary the word is assumed to be correctly classified. If the character combination classified by the network does not appear in the dictionary the Levenshtein distances are calculated for every word in the dictionary. This is in turn used to find the best corresponding word. This is done by using the frequencies per word provided in the dictionary together with the Levenshtein distance. We take e to the power of the distance divided by the word probability. The word probability is the word frequency divided by the total number of occurrences of all words. We then find the word with the smallest value. 
So for example: our classifier found a word with Levenshtein distance 1 to a word with 1000 occurrences in the dictionary. We calculate exp(1) / (1000/total) 
Another method implemented is using a frequncy list of most common Hebrew words rather than the n-gram frequency. 