# import the necessary packages
import numpy as np
import cv2
import os
'''def extractHistogramList(image, label):
	# get the descriptor class initiated
    desc = LocalBinaryPatterns(10, 5)

    histogramList = []
    LabelList = []

    # This mask has the same width and height a the original image and has a default value of 0 (black).
    maskedImage = np.zeros(image.shape[:2], dtype="uint8")
    ########### create imageROIList here ############

    (h, w) = image.shape[:2]

    cellSizeY = h / 4
    cellSizeX = w / 4

    # start in origo
    x = 0
    y = 0
    counterInt = 0

    # 10*10 = 100
    for i in range(4):

        # update this value
        y = cellSizeY * (i)

        x = 0  # it starts at 0 for a new row
        for j in range(4):
            # print "[x] inspecting imageROI %d" % (counterInt)
            counterInt = counterInt + 1

            x = cellSizeX * (j)

            imageROI = image[y: cellSizeY * (i + 1), x:cellSizeX * (j + 1)]

            # print "ystart  " + str(y) + "  yjump  " + str((cellSizeYdir * (i+1)))
            # print "xstart  " + str(x) +  "  xjump  " + str((cellSizeXdir * (j+1)))

            # grayscale and calculate histogram
            grayImageROI = cv2.cvtColor(imageROI, cv2.COLOR_BGR2GRAY)
			if self.preprocessors is not None:
				# loop over the preprocessors and apply each to
				# the image

				for p in self.preprocessors:
					hist = p.describe(image)

            data.append(hist)
            label.append(label)

    return histogramList, LabelList'''
def resizeImage(image):
    (h, w) = image.shape[:2]

    width = 400 #  This "width" is the width of the resize`ed image
    # calculate the ratio of the width and construct the
    # dimensions
    ratio = width / float(w)
    dim = (width, int(h * ratio))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    #resized = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
    return resized
class SimpleDatasetLoader:
	def __init__(self, preprocessors=None):
		# store the image preprocessor
		self.preprocessors = preprocessors

		# if the preprocessors are None, initialize them as an
		# empty list
		if self.preprocessors is None:
			self.preprocessors = []

	def load(self, imagePaths, col, row, verbose=-1):
		# initialize the list of features and labels
		global partial_hist
		data = []
		labels = []


		# loop over the input images
		for (i, imagePath) in enumerate(imagePaths):
			# load the image, convert it to grayscale, and describe it
			image = cv2.imread(imagePath)
			image = resizeImage(image)
			#grayscale
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

			label = imagePath.split(os.path.sep)[-2]
			(h, w) = image.shape[:2]

			cellSizeY = h / col
			cellSizeX = w / row

			# start in origo

			counterInt = 0
			hist = []
			for k in range(col):
				# update this value
				y = cellSizeY * k
				x = 0  # it starts at 0 for a new row
				for j in range(row):
					# print "[x] inspecting imageROI %d" % (counterInt)
					counterInt = counterInt + 1
					x = cellSizeX * j
					imageROI = image[int(y): int(cellSizeY * (k + 1)), int(x): int(cellSizeX * (j + 1))]
					if self.preprocessors is not None:
						# loop over the preprocessors and apply each to
						# the image
						for p in self.preprocessors:
							partial_hist = p.describe(imageROI)
					hist = np.concatenate((hist, np.array(partial_hist)))

			# extract the label from the image path, then update the
			# label and data lists
			if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
				print("[INFO] processed {}/{}".format(i + 1,
					len(imagePaths)))
			data.append(hist)
			labels.append(label)
			# show an update every `verbose` images


		# return a tuple of the data and labels
		return np.array(data), np.array(labels)
#GIAM KICH THUOC TAM ANHHHHHH!!!!! CHIA THANH CAC BLOCK KHAC NHAU

