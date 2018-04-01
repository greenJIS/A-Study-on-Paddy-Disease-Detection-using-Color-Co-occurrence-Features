import glob
import cv2
import math
import numpy as np
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
import h5py
import scipy

from PIL import Image
from scipy import ndimage
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from skimage.filters import threshold_mean
from skimage import data
from skimage.filters import try_all_threshold
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color

from skimage import feature, io
from sklearn import preprocessing
import FeatureExtraction as Features


def helperFunction(arr,index):
    sum = 0.0
    for i in range(4):
        sum = sum + arr.item(index, i)
    sum = sum/4.0
    return sum

arr = []
flag1 = 1
def trainImage(image_list):
    global  arr
    global flag1
    arr.clear()
    for index in range(len(image_list)):
        img = io.imread(image_list[index], as_grey=True)

        infile = cv2.imread(image_list[index])
        infile = infile[:, :, 0]
        hues = (np.array(infile) / 255.) * 179
        outimageHSV = np.array([[[b, 255, 255] for b in a] for a in hues]).astype(int)
        outimageHSV = np.uint8(outimageHSV)

        outimageBGR = cv2.cvtColor(outimageHSV, cv2.COLOR_HSV2BGR)

        rgb = io.imread(image_list[index])
        lab = color.rgb2lab(rgb)

        outimageBGR = lab

        for i in range(outimageBGR.shape[0]):
            for j in range(outimageBGR.shape[1]):
                sum = 0
                for k in range(outimageBGR.shape[2]):
                    sum = sum + outimageBGR[i][j][k]
                sum = sum / (3 * 255)
                if(i<img.shape[0] and j<img.shape[1]):
                    img[i][j] = sum

        S = preprocessing.MinMaxScaler((0, 19)).fit_transform(img).astype(int)
        Grauwertmatrix = feature.greycomatrix(S, [1, 2, 3], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=20,
                                              symmetric=False, normed=True)


        arr.append(feature.greycoprops(Grauwertmatrix, 'contrast'))
        arr.append(feature.greycoprops(Grauwertmatrix, 'correlation'))
        arr.append(feature.greycoprops(Grauwertmatrix, 'homogeneity'))
        arr.append(feature.greycoprops(Grauwertmatrix, 'ASM'))
        arr.append(feature.greycoprops(Grauwertmatrix, 'energy'))
        arr.append(feature.greycoprops(Grauwertmatrix, 'dissimilarity'))
        arr.append(Features.sumOfSquares(Grauwertmatrix))
        arr.append(Features.sumAverage(Grauwertmatrix))
        arr.append(Features.sumVariance(Grauwertmatrix))
        arr.append(Features.Entropy(Grauwertmatrix))
        arr.append(Features.sumEntropy(Grauwertmatrix))
        arr.append(Features.differenceVariance(Grauwertmatrix))
        arr.append(Features.differenceEntropy(Grauwertmatrix))
        arr.append(Features.informationMeasureOfCorelation1(Grauwertmatrix))
        arr.append(Features.informationMeasureOfCorelation2(Grauwertmatrix))
    flag1 = 1

arr1 = []

flag = 1
def testImage(image_list):
    global arr1
    global flag
    arr1.clear()
    for index in range(len(image_list)):
        img = io.imread(image_list[index], as_grey=True)

        infile = cv2.imread(image_list[index])
        infile = infile[:, :, 0]
        hues = (np.array(infile) / 255.) * 179
        outimageHSV = np.array([[[b, 255, 255] for b in a] for a in hues]).astype(int)
        outimageHSV = np.uint8(outimageHSV)

        outimageBGR = cv2.cvtColor(outimageHSV, cv2.COLOR_HSV2BGR)


        rgb = io.imread(image_list[index])
        lab = color.rgb2lab(rgb)

        outimageBGR = lab

        for i in range(outimageBGR.shape[0]):
            for j in range(outimageBGR.shape[1]):
                sum = 0
                for k in range(outimageBGR.shape[2]):
                    sum = sum + outimageBGR[i][j][k]
                sum = sum / (3 * 255)
                if (i < img.shape[0] and j < img.shape[1]):
                    img[i][j] = sum


        S = preprocessing.MinMaxScaler((0, 19)).fit_transform(img).astype(int)
        Grauwertmatrix = feature.greycomatrix(S, [1, 2, 3], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=20,
                                              symmetric=False, normed=True)

        arr1.append(feature.greycoprops(Grauwertmatrix, 'contrast'))
        arr1.append(feature.greycoprops(Grauwertmatrix, 'correlation'))
        arr1.append(feature.greycoprops(Grauwertmatrix, 'homogeneity'))
        arr1.append(feature.greycoprops(Grauwertmatrix, 'ASM'))
        arr1.append(feature.greycoprops(Grauwertmatrix, 'energy'))
        arr1.append(feature.greycoprops(Grauwertmatrix, 'dissimilarity'))
        arr1.append(Features.sumOfSquares(Grauwertmatrix))
        arr1.append(Features.sumAverage(Grauwertmatrix))
        arr1.append(Features.sumVariance(Grauwertmatrix))
        arr1.append(Features.Entropy(Grauwertmatrix))
        arr1.append(Features.sumEntropy(Grauwertmatrix))
        arr1.append(Features.differenceVariance(Grauwertmatrix))
        arr1.append(Features.differenceEntropy(Grauwertmatrix))
        arr1.append(Features.informationMeasureOfCorelation1(Grauwertmatrix))
        arr1.append(Features.informationMeasureOfCorelation2(Grauwertmatrix))
    flag = 1

#print("After applying GLCM the features are : ")
def GLCM(Matrix,index,mask,flag):
    global  arr
    global  arr1
    if(flag==0):#training
        id = 0
        for i in range(3):
            for j in range(len(arr)):
                if( (mask & (1<<j)) == 0 ):
                    continue
                ret = helperFunction(arr[index*15+j],i)
                Matrix[id][index] = ret
                id = id + 1
    else :#testing
        id = 0
        for i in range(3):
            for j in range(len(arr)):
                if ((mask & (1 << j)) == 0):
                    continue
                ret = helperFunction(arr1[index*15+j], i)
                Matrix[id][index] = ret
                id = id + 1
