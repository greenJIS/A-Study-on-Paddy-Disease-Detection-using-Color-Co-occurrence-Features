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

from skimage import feature, io
from sklearn import preprocessing
import main as rem



X = []
XX = []
y = []
yy = []
image_list = []
Image_list = []


for file in glob.glob('./Narrow/*.*'):
    img = cv2.resize(cv2.imread(file),(64,64),interpolation=cv2.INTER_CUBIC)
    X.append(img)
    image_list.append(file)
    Image_list.append(file)
    y.append(0)
    XX.append(img)
    yy.append(0)

for file in glob.glob('./Brown/*.*'):
    img = cv2.resize(cv2.imread(file),(64,64),interpolation=cv2.INTER_CUBIC)
    X.append(img)
    image_list.append(file)
    Image_list.append(file)
    y.append(2)
    XX.append(img)
    yy.append(2)


for file in glob.glob('./Paddy/*.*'):
    img = cv2.resize(cv2.imread(file),(64,64),interpolation=cv2.INTER_CUBIC)
    X.append(img)
    image_list.append(file)
    Image_list.append(file)
    y.append(1)
    XX.append(img)
    yy.append(1)

for file in glob.glob('./Other/*.*'):
    img = cv2.resize(cv2.imread(file),(64,64),interpolation=cv2.INTER_CUBIC)
    X.append(img)
    image_list.append(file)
    Image_list.append(file)
    y.append(3)
    XX.append(img)
    yy.append(3)

X = np.array(X)
y = np.array(y)

train_set_x_orig = X
train_set_y = y


X1 = []
y1 = []
image_list1 = []


for file in glob.glob('./Narrow/Testing/*.*'):
    img = cv2.resize(cv2.imread(file),(64,64),interpolation=cv2.INTER_CUBIC)
    X1.append(img)
    image_list1.append(file)
    Image_list.append(file)
    y1.append(0)
    XX.append(img)
    yy.append(0)

for file in glob.glob('./Brown/Testing/*.*'):
    img = cv2.resize(cv2.imread(file),(64,64),interpolation=cv2.INTER_CUBIC)
    X1.append(img)
    image_list1.append(file)
    Image_list.append(file)
    y1.append(2)
    XX.append(img)
    yy.append(2)


for file in glob.glob('./Paddy/Testing/*.*'):
    img = cv2.resize(cv2.imread(file),(64,64),interpolation=cv2.INTER_CUBIC)
    X1.append(img)
    image_list1.append(file)
    Image_list.append(file)
    y1.append(1)
    XX.append(img)
    yy.append(1)


X1 = np.array(X1)
y1 = np.array(y1)

test_set_x_orig = X1
test_set_y = y1

rem.trainImage(image_list)
rem.testImage(image_list1)


Number_of_features = 0
flag = 0
def featuresFunction(mask):
    cnt = 0
   # print (mask)
    for i in range(15):
        if( (mask & (1<<i)) ):
            cnt = cnt + 1
    #print(cnt)
    assert cnt>0
    global Number_of_features
    Number_of_features = cnt * 3

    Train_Matrix = np.zeros(shape=(Number_of_features,len(image_list)))

    for i in range(len(image_list)):
        rem.GLCM(Train_Matrix,i,mask,0)

    Test_Matrix = np.zeros(shape=(Number_of_features,len(image_list1)))

    for i in range(len(image_list1)):
        rem.GLCM(Test_Matrix, i,mask,1)
    global  flag

    return Train_Matrix,Test_Matrix




## Load input output from a text file

#print(Test_Matrix[0][0])
#print(Test_Matrix[0][1])
#np.savetxt('Training_Data.txt', Test_Matrix , fmt='%2.9e',newline='\n\n')

#Check_Matrix = np.zeros(shape=(Number_of_features,len(image_list)))
#input = np.loadtxt("Training_Data.txt")
#print(input[0][0])
#print(input[0][1])
