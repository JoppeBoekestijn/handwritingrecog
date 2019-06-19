# Main file
We start with asking for the folder where the input images are located. Then we check that folder and see which files are inside. We try to find all jpg images. Nestor stated there should be 20 images, so we check for that.
We then use this list to load each image into our recognition system and let it process the image. After the pre-processing, line segmentation, classification and the language model we output the classified characters to a text file and save that under the same name as the input image. 


# Pre processing file



# Line segmentation file
The line segmentation receives the binarized image and a mask from the pre-processing stage. 
The first step in the line segmentation is to transform the image to a feature space where we can derive information from about where in the image the lines are. This is achieved by summing all ink pixels per horizontal row of one pixel height. What we then get is a histogram-like graph with pixel densities. This would provide us with decent results on a regular, rectangular a4 paper. But, we are working with fragments of the dead sea scrolls. These are very irregular shaped pieces of paper with differing widths. Therefore we add a normalization to the summation. This is done by dividing the number of found ink pixels through the paper width at that position.
We then try to find the most prominent peaks by examining the generated histogram. This is done using a peak finding algorithm with three parameters: width, distance and height. Width resembles the width of the peak and disregards every peak with a width lower than 5. Distance is the minimum distance the peaks should appear from each other. This is computed by dividing the height of the image through the maximum number of lines that appeared in the provided dataset plus a small buffer. The height parameter states the minimum height that the normalized peak should have. We found that it should not be smaller than 1/5th of the highest peak. 
The coordinates of the most prominent peaks are used to determine the slicing coordinates. These are calculated by finding the exact middle distance between two peaks. So two peak coordinates are added and divided by two. This method was created under the, rather strong, assumption that each new line in the image is spaced to a constant ratio. 
These three steps leaves us with a number of slices ready to be processed in the next step.
 

# Binarization file



# classification file



# Language model file
The word classified with the neural network is sent to the language model. This model first checks if the word occurs in the provided ngram dictionary. If it is mentioned in the dictionary the word is assumed to be correctly classified. If the character combination classified by the network does not appear in the dictionary the Levenshtein distances are calculated for every word in the dictionary. This is in turn used to find the best corresponding word. This is done by using the frequencies per word provided in the dictionary together with the Levenshtein distance. We take e to the power of the distance divided by the word probability. The word probability is the word frequency divided by the total number of occurrences of all words. We then find the word with the smallest value. 
So for example: our classifier found a word with Levenshtein distance 1 to a word with 1000 occurrences in the dictionary. We calculate exp(1) / (1000/total) 