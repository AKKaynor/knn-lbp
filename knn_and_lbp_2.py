# USAGE
# python knn.py --dataset ../datasets/animals

# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from pyimagesearch.LocalBinaryPattern import LocalBinaryPatterns
from pyimagesearch.LocalBinaryPattern import SimpleDatasetLoader
from imutils import paths

import argparse
import cv2
import os
import numpy as np
import pandas as pd
#chisquare distance
def chisquare(x, y):
	return 0.5 * np.sum((x - y) ** 2 / (x + y + 1e-6))
#export the classification to csv file
def classification_report_csv(rp):
	report_data = []
	lines = rp.split('\n')
	for line in lines[2:5]:
		row = {}
		row_data = line.split('      ')
		row['class'] = row_data[1]
		row['precision'] = (row_data[2])
		row['recall'] = (row_data[3])
		row['f1_score'] = (row_data[4])
		row['support'] = (row_data[5])
		report_data.append(row)
	row = {}
	row_data = lines[6].split('      ')
	row['class'] = row_data[0]
	row['precision'] = (row_data[1])
	row['recall'] = (row_data[2])
	row['f1_score'] = (row_data[3])
	row['support'] = (row_data[4])
	report_data.append(row)
	data_frame = pd.DataFrame.from_dict(report_data)
	data_frame.to_csv('classification_report.csv', index = False)
#initial some parameter, easier to change in the future
numPoints = 8
radius = 1
col = 4
#define image path
imagePaths = list(paths.list_images('../datasets/animals'))
#define the features that will be used for the data
desc = LocalBinaryPatterns(numPoints, radius)
sdl = SimpleDatasetLoader(preprocessors=[desc])
# loop over the training images
#load the data
(data, labels) = sdl.load(imagePaths, col, col, verbose=50)
print("numPoints=",numPoints, "radius=", radius, "column=", col)
# show some information on memory consumption of the images
print("[INFO] features matrix: {:.1f}MB".format(
	data.nbytes / (1024 * 1000.0)))
'''Step #2'''
# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing X=data, Y=label
(trainX, testX, trainY, testY) = train_test_split(data, labels,	test_size=0.25, random_state=42)
'''Step #3, 4'''
# train and evaluate a k-NN classifier on the raw pixel intensities
print("[INFO] evaluating k-NN classifier...")
#1. dung kNN + chi-square, 2.LinearSVMinearSVM
#model = KNeighborsClassifier(n_neighbors=1, p=1, weights='distance', metric='euclidean', n_jobs=-1)
#model = KNeighborsClassifier(n_neighbors=1, p=1, weights='distance', metric=chisquare, n_jobs=-1)
model = LinearSVC(C=101, class_weight='balanced')
model.fit(trainX, trainY)
report = classification_report(testY, model.predict(testX),
									target_names=le.classes_)
print(report)
classification_report_csv(report)

filename = 'knn_model_lbp_y.sav'

# some time later...

# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.score(testX, testY)
print(result)

'''
imgPath = input("Input Image Path: ")
img = cv2.imread('pyimagesearch/' + imgPath)
tmp = img
img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_LINEAR)
img = img.reshape((1, 3072))
lbl =loaded_model.predict(img)
print(loaded_model.kneighbors(img))
print(lbl)
font = cv2.FONT_HERSHEY_SIMPLEX
if lbl == 0:
	cv2.putText(tmp, "This is a cat!", (0, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
	cv2.imshow('img', tmp)
elif lbl == 1:
	cv2.putText(tmp, "This is a dog!", (0, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
	cv2.imshow('img', tmp)
elif lbl == 2:
	cv2.putText(tmp, "This is a panda!", (0, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
	cv2.imshow('img', tmp)
cv2.waitKey(0)
'''
