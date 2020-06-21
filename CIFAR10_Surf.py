import cv2
from skimage import io
import numpy as np
import csv
import pickle

def load_cfar10_batch(path):
    with open(path, mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels
surf_extractor = cv2.xfeatures2d.SURF_create()
features = []
labels=[]


images,class_labels=load_cfar10_batch("/Users/daggubatisirichandana/PycharmProjects/MLTechniques/_VisualRecognition/Assignment2/cifar_10/data_batch_1")
index=[]
id=0
for img in images :
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    keypoint, descriptor = surf_extractor.detectAndCompute(gray, None)
    if (descriptor is not None):
        vectorized = np.float32(descriptor.reshape(-1,1))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, centers = cv2.kmeans(vectorized, 30, None, criteria, 10, cv2.KMEANS_PP_CENTERS)#clusters and iterations
        freq = np.histogram(label,bins=range(0,31))[0].tolist()
        features+=[freq]
    else:
        index+=[id]
    id+=1
for ele in sorted(index, reverse = True):
	del class_labels[ele]
labels=class_labels

images,class_labels=load_cfar10_batch("/Users/daggubatisirichandana/PycharmProjects/MLTechniques/_VisualRecognition/Assignment2/cifar_10/data_batch_2")
index=[]
id=0
for img in images :
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    keypoint, descriptor = surf_extractor.detectAndCompute(gray, None)
    if (descriptor is not None):
        vectorized = np.float32(descriptor.reshape(-1,1))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, centers = cv2.kmeans(vectorized, 30, None, criteria, 10, cv2.KMEANS_PP_CENTERS)#clusters and iterations
        freq = np.histogram(label,bins=range(0,31))[0].tolist()
        features+=[freq]
    else:
        index+=[id]
    id+=1
for ele in sorted(index, reverse = True):
	del class_labels[ele]
labels=labels+class_labels

images,class_labels=load_cfar10_batch("/Users/daggubatisirichandana/PycharmProjects/MLTechniques/_VisualRecognition/Assignment2/cifar_10/data_batch_3")
index=[]
id=0
for img in images :
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    keypoint, descriptor = surf_extractor.detectAndCompute(gray, None)
    if (descriptor is not None):
        vectorized = np.float32(descriptor.reshape(-1,1))
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, centers = cv2.kmeans(vectorized, 30, None, criteria, 10, cv2.KMEANS_PP_CENTERS)#clusters and iterations
        freq = np.histogram(label,bins=range(0,31))[0].tolist()
        features+=[freq]
    else:
        index+=[id]
    id+=1
for ele in sorted(index, reverse = True):
	del class_labels[ele]
labels=labels+class_labels

images,class_labels=load_cfar10_batch("/Users/daggubatisirichandana/PycharmProjects/MLTechniques/_VisualRecognition/Assignment2/cifar_10/data_batch_4")
index=[]
id=0
for img in images :
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    keypoint, descriptor = surf_extractor.detectAndCompute(gray, None)
    if (descriptor is not None):
        vectorized = np.float32(descriptor.reshape(-1,1))
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, centers = cv2.kmeans(vectorized, 30, None, criteria, 10, cv2.KMEANS_PP_CENTERS)#clusters and iterations
        freq = np.histogram(label,bins=range(0,31))[0].tolist()
        features+=[freq]
    else:
        index+=[id]
    id+=1
for ele in sorted(index, reverse = True):
	del class_labels[ele]
labels=labels+class_labels

images,class_labels=load_cfar10_batch("/Users/daggubatisirichandana/PycharmProjects/MLTechniques/_VisualRecognition/Assignment2/cifar_10/data_batch_5")
index=[]
id=0
for img in images :
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    keypoint, descriptor = surf_extractor.detectAndCompute(gray, None)
    if (descriptor is not None):
        vectorized = np.float32(descriptor.reshape(-1,1))
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, centers = cv2.kmeans(vectorized, 30, None, criteria, 10, cv2.KMEANS_PP_CENTERS)#clusters and iterations
        freq = np.histogram(label,bins=range(0,31))[0].tolist()
        features+=[freq]
    else:
        index+=[id]
    id+=1
for ele in sorted(index, reverse = True):
	del class_labels[ele]
labels=labels+class_labels

features+=[labels]
with open('cifar_surf_train.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerows(features)


features = []
labels=[]
images,class_labels=load_cfar10_batch("/Users/daggubatisirichandana/PycharmProjects/MLTechniques/_VisualRecognition/Assignment2/cifar_10/test_batch")
index=[]
id=0
for img in images :
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    keypoint, descriptor = surf_extractor.detectAndCompute(gray, None)
    if (descriptor is not None):
        print(len(descriptor[0]))
        vectorized = np.float32(descriptor.reshape(-1,1))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, centers = cv2.kmeans(vectorized, 30, None, criteria, 10, cv2.KMEANS_PP_CENTERS)#clusters and iterations
        freq = np.histogram(label,bins=range(0,31))[0].tolist()
        features+=[freq]
    else:
        index+=[id]
    id+=1
for ele in sorted(index, reverse = True):
	del class_labels[ele]


features+=[class_labels]
with open('cifar_surf_test.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerows(features)
