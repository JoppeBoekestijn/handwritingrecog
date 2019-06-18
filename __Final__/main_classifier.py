import sys
import os
import lineSegmentation as ls


def main(argv):
	source_path = str(argv[1])
	source_dir = source_path.rsplit('/',1)[1]
	print('\n')
	print('Classifier started in folder {}'.format(source_dir))
	assert os.path.exists(argv[1]), "The source directory {} does not exist".format(source_dir)
	
	sourceFiles = os.listdir(source_dir)
	
	sourceImages = []
	for file in sourceFiles:
		if file.endswith('.jpg'):
			sourceImages.append(file)

	if len(sourceImages) == 20:
		print('Started with processing of {} image files with the .jpg extension'.format(len(sourceImages)))
	else:
		print('Something went wrong, the classifier found {} .jpg images in the test folder. There should be 20.'.format(len(sourceImages)))
		print('But continuing anyway...')
		print('Started with processing of {} image files with the .jpg extension'.format(len(sourceImages)))
	
	# Main loop for classification per image
	for file in sourceImages:
		imageName = file.rsplit('.',1)[0]
		imagePath = source_path+'/'+file
		print('Processing file {}'.format(imageName))
		
		# Binarization
		
		# Line segmentation
		histogram = ls.getHist(output_hole_removal, mask)
		
		slices = ls.findSlices(histogram)
		
		slicesList = ls.sliceImg(inputimg, original, mask, slices)
		# Hole removal, slices segmentation
		
		# Classification
		lineList = []
		# Language model
		
		
		
		# Writing to .txt file
		textf = source_path + '/' + imageName + '.txt'
		#textfile = open(textf, 'w')
		#textfile.writelines(lineList)
		#textfile.close()
		print('Converted image file {} to text, saved to {}'.format(imageName,textf))
		


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Please input the path to the folder containing the 20 test images"

    main(sys.argv)