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


def sumOfSquares(P):
    a = np.matrix('0.0 0.0 0.0 0.0;0.0 0.0 0.0 0.0;0.0 0.0 0.0 0.0')
    for k in range ( P.shape[2] ):
        for l in range ( P.shape[3] ):
            meu = 0.0
            for i in range ( P.shape[0] ):
                for j in range ( P.shape[1] ):
                    meu = meu + P[i][j][k][l]
            meu = meu / (P.shape[0]*P.shape[0])
            #print(meu)
            sum = 0.0
            for i in range ( P.shape[0] ):
                for j in range ( P.shape[1] ):
                    sum = sum + (i-meu+1)*(i-meu+1)*P[i][j][k][l]
            a[k,l] = sum
    return a

def sumAverage(P):
    a = np.matrix('0.0 0.0 0.0 0.0;0.0 0.0 0.0 0.0;0.0 0.0 0.0 0.0')
    for k in range(P.shape[2]):
        for l in range(P.shape[3]):
            sum = 0.0
            arr = []
            for i in range (2*P.shape[0]+1):
                arr.append(0.0)
            for i in range (P.shape[0]):
                for j in range (P.shape[1]):
                    arr[ i+j+2 ] = arr[i+j+2]+P[i][j][k][l]
            for i in range(P.shape[0]*2+1):
                if( i<2 ):
                    continue
                sum = sum + arr[i] * i
            a[k, l] = sum
    return a

def Entropy(P):
    a = np.matrix('0.0 0.0 0.0 0.0;0.0 0.0 0.0 0.0;0.0 0.0 0.0 0.0')
    for k in range(P.shape[2]):
        for l in range(P.shape[3]):
            sum = 0.0
            for i in range(P.shape[0]):
                for j in range(P.shape[1]):
                    if(P[i][j][k][l]==0.0):
                        continue
                    sum = sum + P[i][j][k][l] * math.log10(P[i][j][k][l])
            a[k, l] = -1.0*sum
    return a

def sumEntropy(P):
    a = np.matrix('0.0 0.0 0.0 0.0;0.0 0.0 0.0 0.0;0.0 0.0 0.0 0.0')
    for k in range(P.shape[2]):
        for l in range(P.shape[3]):
            arr = []
            for i in range(2 * P.shape[0] + 1):
                arr.append(0.0)
            for i in range(P.shape[0]):
                for j in range(P.shape[1]):
                    arr[i + j + 2] = arr[i + j + 2] + P[i][j][k][l]
            sum = 0.0
            for i in range(P.shape[0]*2+1):
                if( i<2 or arr[i]==0.0 ):
                    continue
                sum = sum + arr[i] * math.log10(arr[i])
            sum = sum * -1.0
            a[k, l] = sum
    return a

def sumVariance(P):
    a = np.matrix('0.0 0.0 0.0 0.0;0.0 0.0 0.0 0.0;0.0 0.0 0.0 0.0')
    for k in range(P.shape[2]):
        for l in range(P.shape[3]):
            arr = []
            for i in range(2 * P.shape[0] + 1):
                arr.append(0.0)
            for i in range(P.shape[0]):
                for j in range(P.shape[1]):
                    arr[i + j + 2] = arr[i + j + 2] + P[i][j][k][l]
            f = 0.0
            for i in range(P.shape[0] * 2 + 1):
                if (i < 2 or arr[i] == 0.0):
                    continue
                f = f + arr[i] * math.log10(arr[i])
            f = f * -1.0

            sum = 0.0
            for i in range(P.shape[0] * 2 + 1):
                if (i < 2):
                    continue
                sum = sum +  (i-f)*(i-f)*arr[i]
            a[k, l] = sum
    return a

def differenceEntropy(P):
    a = np.matrix('0.0 0.0 0.0 0.0;0.0 0.0 0.0 0.0;0.0 0.0 0.0 0.0')
    for k in range(P.shape[2]):
        for l in range(P.shape[3]):
            arr = []
            for i in range(P.shape[0]):
                arr.append(0.0)
            for i in range(P.shape[0]):
                for j in range(P.shape[1]):
                    arr[abs(i-j)] = arr[abs(i-j)] + P[i][j][k][l]
            sum = 0.0
            for i in range (P.shape[0]):
                if(arr[i]==0.0):
                    continue
                sum = sum + arr[i]*math.log10(arr[i])
            sum = sum * -1.0
            a[k, l] = sum
    return a

def differenceVariance(P):
    a = np.matrix('0.0 0.0 0.0 0.0;0.0 0.0 0.0 0.0;0.0 0.0 0.0 0.0')
    for k in range(P.shape[2]):
        for l in range(P.shape[3]):
            sum = 0.0
            arr = []
            for i in range(P.shape[0]):
                arr.append(0.0)
            for i in range(P.shape[0]):
                for j in range(P.shape[1]):
                    arr[ abs(i-j) ] = arr[ abs(i-j) ] + P[i][j][k][l]
            mean = 0.0
            for i in range(P.shape[0]):
                mean = mean + arr[i]
            mean = mean / P.shape[0]
            for i in range(P.shape[0]):
                sum = sum + ( (mean-arr[i])*(mean-arr[i]) )
            sum = sum / P.shape[0]
            a[k, l] = sum
    return a



def informationMeasureOfCorelation1(P):
    a = np.matrix('0.0 0.0 0.0 0.0;0.0 0.0 0.0 0.0;0.0 0.0 0.0 0.0')
    for k in range(P.shape[2]):
        for l in range(P.shape[3]):
            HXY = 0.0
            for i in range (P.shape[0]):
                for j in range (P.shape[1]):
                    if( P[i][j][k][l]==0.0 ):
                        continue
                    HXY = HXY + P[i][j][k][l]*math.log10(P[i][j][k][l]);
            HXY = HXY * -1
            Px = []
            Py = []
            for i in range( P.shape[0] ):
                Px.append(0.0)
                Py.append(0.0)
            for i in range ( P.shape[0]):
                for j in range (P.shape[1]):
                    Px[i] = Px[i] + P[i][j][k][l]
                    Py[j] = Py[j] + P[i][j][k][l]
            HXY1 = 0.0
            for i in range (P.shape[0]):
                for j in range(P.shape[1]):
                    if( Px[i] * Py[j] == 0.0 ):
                        continue
                    HXY1 = HXY1 + P[i][j][k][l] * math.log10( Px[i] * Py[j] )
            HXY1 = HXY1 * -1
            HX = 0.0
            HY = 0.0
            for i in range (P.shape[0]):
                if(Px[i]==0.0):
                    continue
                HX = HX + Px[i] * math.log10(Px[i])
            for i in range(P.shape[0]):
                if (Py[i] == 0.0):
                    continue
                HY = HY + Py[i] * math.log10(Py[i])
            HX = HX * -1.0
            HY = HY * -1.0
            temp = max(HX,HY)
            if(temp==0.0):
                a[k, l] = ( HXY - HXY1  )
                continue
            a[k, l] = ( HXY - HXY1  ) / temp
    return a


def informationMeasureOfCorelation2(P):
    a = np.matrix('0.0 0.0 0.0 0.0;0.0 0.0 0.0 0.0;0.0 0.0 0.0 0.0')
    for k in range(P.shape[2]):
        for l in range(P.shape[3]):
            HXY = 0.0
            for i in range(P.shape[0]):
                for j in range(P.shape[1]):
                    if (P[i][j][k][l] == 0.0):
                        continue
                    HXY = HXY + P[i][j][k][l] * math.log10(P[i][j][k][l]);
            HXY = HXY * -1
            Px = []
            Py = []
            for i in range(P.shape[0]):
                Px.append(0.0)
                Py.append(0.0)
            for i in range(P.shape[0]):
                for j in range(P.shape[1]):
                    Px[i] = Px[i] + P[i][j][k][l]
                    Py[j] = Py[j] + P[i][j][k][l]
            HXY2 = 0.0
            for i in range(P.shape[0]):
                for j in range(P.shape[1]):
                    if (Px[i] * Py[j] == 0.0):
                        continue
                    HXY2 = HXY2 + Px[i] * Py[j] * math.log10(Px[i] * Py[j])
            HXY2 = HXY2 * -1
            value = 1.0 - math.exp( -2.0 * (HXY2-HXY) )
            a[k, l] = math.sqrt(value)
    return a



