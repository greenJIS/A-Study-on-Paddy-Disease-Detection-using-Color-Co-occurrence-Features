import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os
import glob
import numpy as np
from sklearn.metrics import confusion_matrix
import random
from skimage import io, color
import DataProcessing as load
import main as rem
from tensorflow.examples.tutorials.mnist import input_data
from xlwt import Workbook
import xlrd

import matplotlib.patches as patches
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw



font = {'family': 'serif',
        'color':  'blue',
        'weight': 'normal',
        'size': 10,
        }



def rgbCalculation(img):
    minR = 255
    maxR = 0
    minG = 255
    maxG = 0
    minB = 255
    maxB = 0
    #img = cv2.imread(imagename)
    #img = io.imread(imagename, as_grey=False)
    #print(img.shape)
    #plt.imshow(img)
    #plt.show()
    for i in range (img.shape[0]):
        for j in range(img.shape[1]):
            for k in range (img.shape[2]):
                if(k==0):
                    minR = min(minR,img[i][j][k])
                    maxR = max(maxR,img[i][j][k])
                elif(k==1):
                    minG = min(minG, img[i][j][k])
                    maxG = max(maxG, img[i][j][k])
                else:
                    minB = min(minB, img[i][j][k])
                    maxB = max(maxB, img[i][j][k])
    # print(minR)
    # print(maxR)
    # print(minG)
    # print(maxG)
    # print(minB)
    # print(maxB)
    return ( minR>=93 and minR<=211 and maxR>=93 and maxR<=211 and
             minG >= 142 and minG <= 222 and maxG >= 142 and maxG <= 222 and
             minB >= 64 and minB <= 155 and maxB >= 64 and maxB <= 155 )


n = len(load.XX)
Number_of_features = 0
flag = 1
normalizeValue = 0.0

def featuresFunction(mask,image_list,flagvalue):
    cnt = 0
    for i in range(15):
        if( (mask & (1<<i)) ):
            cnt = cnt + 1
    #print(cnt)
    assert cnt>0
    global Number_of_features
    Number_of_features = cnt * 3
    Train_Matrix = np.zeros(shape=(Number_of_features,len(image_list)))
    for i in range(len(image_list)):
        rem.GLCM(Train_Matrix,i,mask,flagvalue)
    # wb = Workbook()
    # sheet1 = wb.add_sheet('Train Matrix')
    # for i in range(Train_Matrix.shape[0]):
    #     for j in range(Train_Matrix.shape[1]):
    #         sheet1.write(i, j, Train_Matrix[i][j])
    # wb.save('Feature Data.xls')
    return Train_Matrix




#############################################################################################
## Neural Network Code
n_nodes_hl1 = 50
n_nodes_hl2 = 50
n_nodes_hl3 = 50

n_classes = 4
batch_size = 100

# input feature size = 28x28 pixels = 784
x_var = tf.placeholder('float', [None, 15])
y_var = tf.placeholder('float')


def neural_network_model(data):
    # input_data * weights + biases
    hidden_l1 = {'weights': tf.Variable(tf.random_normal([15, n_nodes_hl1])),
                 'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_l2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                 'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_l3 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                 'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_l = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'biases': tf.Variable(tf.random_normal([n_classes]))}


    l1 = tf.add(tf.matmul(data, hidden_l1['weights']), hidden_l1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_l2['weights']), hidden_l2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_l3['weights']), hidden_l3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_l['weights']), output_l['biases'])
    return output


prediction = neural_network_model(x_var)

def estimatedPrediction(test_set_x,One_hot_matrix_test):
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_var, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    acc = accuracy.eval({x_var: test_set_x.transpose(), y_var: One_hot_matrix_test})
    #print('Accuracy rate:', acc)
    return acc


def classChecking(file,value):
    image_list1 = []
    y = []
    image_list1.append(file)
    y.append(value)
    test_set_y = np.array(y)
    rem.testImage(image_list1)
    mask = 24604
    test_set_x_flatten = featuresFunction(mask, image_list1,1)
    test_set_x = test_set_x_flatten / normalizeValue
    ##############
    num_classes = 4
    One_hot_matrix_test = np.zeros(shape=(len(test_set_y), num_classes))
    for i in range(len(test_set_y)):
        One_hot_matrix_test[i][test_set_y[i]] = 1
    return estimatedPrediction(test_set_x,One_hot_matrix_test)>0.0

# def Testing():
#     for file in glob.glob('./TestImage/*.*'):
#         img = io.imread(file, as_grey=False)
#         # 0 ---  means Narrow Brown Spot Disease
#         # 1 ---  means Paddy Blast Disease
#         # 2 ---  means Brown Spot Disease
#
#         if (classChecking(file,2)):
#             plt.imshow(img)
#             plt.title("Brown Spot")
#             plt.show()
#         elif (classChecking(file,0)):
#             plt.imshow(img)
#             plt.title("Narrow Brown Spot")
#             plt.show()
#         elif(classChecking(file,1)):
#             plt.imshow(img)
#             plt.title("Paddy Blast")
#             plt.show()
#         else:
#             plt.imshow(img)
#             plt.title("Other")
#             plt.show()

def Testing(str):
    for file in glob.glob(str):
        img = io.imread(file, as_grey=False)
        # 0 ---  means Narrow Brown Spot Disease
        # 1 ---  means Paddy Blast Disease
        # 2 ---  means Brown Spot Disease

        if (classChecking(file,2)):
            plt.imshow(img)
            plt.title("Brown Spot")
            plt.show()
        elif (classChecking(file,0)):
            plt.imshow(img)
            plt.title("Narrow Brown Spot")
            plt.show()
        elif(classChecking(file,1)):
            plt.imshow(img)
            plt.title("Paddy Blast")
            plt.show()
        else:
            plt.imshow(img)
            plt.title("Other")
            plt.show()


def fun():
    for f in glob.glob('./Testing/*.*'):
        os.remove(f)
    for f in glob.glob('./Picture/*.*'):
        os.remove(f)
    lvlIndex = 0
    for file in glob.glob('./Picture1/*.*'):
        img = cv2.imread(file, 1)
        xxxx = 258
        for ii in range(3):
            lvlIndex += 1
            path = './Picture/' + str(lvlIndex) + '.jpg'
            cv2.imwrite(path, cv2.resize(img, ( int(xxxx), int(xxxx) ), interpolation=cv2.INTER_LINEAR))
            xxxx /= 2

    indd = 0
    imageNo = 0
    for file in glob.glob('./Picture/*.*'):
        img1 = io.imread(file, as_grey=False)
        im = np.array(Image.open(file), dtype=np.uint8)
        fig, ax = plt.subplots(1)
        ax.imshow(im)
        img = cv2.imread(file, 1)
        # newimg = cv2.imread('./Waste/img.jpg', 1)
        newimg = cv2.resize( img , (64, 64), interpolation=cv2.INTER_LINEAR)
        # newimg = io.imread('./Waste/img.jpg', as_grey=False)
        flag = 0
        # 0 ---  means Narrow Brown Spot Disease
        # 1 ---  means Paddy Blast Disease
        # 2 ---  means Brown Spot Disease
        ClassZero = 0
        ClassOne = 0
        ClassTwo = 0
        index = 0
        tmpIndex = 0
        imageNo += 1
        for i in range(0, img.shape[0], 25):
            for j in range(0, img.shape[1], 25):
                if (i + 64 <= img.shape[0] and j + 64 <= img.shape[1]):
                    for k in range(i, i + 64):
                        for l in range(j, j + 64):
                            for p in range(3):
                                newimg[k - i][l - j][p] = img[k][l][p]
                    if (rgbCalculation(img1) == False):
                        flag = 1
                        tmpIndex += 1
                        path = './Testing/Level' + str(imageNo) + '_' + str(tmpIndex) + '.jpg'
                        cv2.imwrite(path, cv2.resize(newimg, (64, 64), interpolation=cv2.INTER_LINEAR) )
                        if (classChecking(path,0)):
                            rect = patches.Rectangle((i, j), 64, 64, linewidth=1, edgecolor='r', facecolor='none')
                            ax.add_patch(rect)
                            plt.text(i, j, "0", font)
                            ClassZero += 1
                        else:
                            if (classChecking(path,2)):
                                rect = patches.Rectangle((i, j), 64, 64, linewidth=1, edgecolor='r', facecolor='none')
                                ax.add_patch(rect)
                                plt.text(i, j, "2", font)
                                ClassTwo += 1
                            elif (classChecking(path,1)):
                                rect = patches.Rectangle((i, j), 64, 64, linewidth=1, edgecolor='r', facecolor='none')
                                ax.add_patch(rect)
                                plt.text(i, j, "1", font)
                                ClassOne += 1
                            else:
                                print("None")


        print(ClassZero)
        print(ClassOne)
        print(ClassTwo)
        if (flag == 0):
            plt.title("Normal Leaf")
        else:
            if (ClassTwo >= ClassOne and ClassTwo >= ClassZero):
                plt.title("Brown Spot")
            elif (ClassZero >= ClassTwo and ClassZero >= ClassOne):
                plt.title("Narrow Brown Spot")
            else:
                plt.title("Paddy Blast")
        plt.show()
        indd += 1
        path2 = './Full/' + str(indd) + '.jpg'
        fig.savefig(path2)



def train_neural_network(x_var,train_set_x,One_hot_matrix_train): #,train_set_x,One_hot_matrix_train,test_set_x,One_hot_matrix_test
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_var))  # v1.0 changes
    # optimizer = tf.train.AdamOptimizer().minimize(cost)
    epochs_no = 90001
    epoch_x = train_set_x.transpose()
    epoch_y = One_hot_matrix_train
    saver = tf.train.Saver()
    rem = .98
    cst = 12629
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())  # v1.0 changes
        # # training
        # for epoch in range(epochs_no):
        #     _, c = sess.run([optimizer, cost], feed_dict={x_var: epoch_x, y_var: epoch_y})
        #
        #     if(epoch%3000==0):
        #         print('Epoch', epoch, 'completed out of', epochs_no, 'loss:', c)
        #         acc = estimatedPrediction(train_set_x, One_hot_matrix_train)
        #         if(acc>rem):
        #             rem = acc
        #             cst = c
        #             save_path = saver.save(sess, "my_net/save_net.ckpt")
        #         elif(acc==rem and c<cst):
        #             cst = c
        #             save_path = saver.save(sess, "my_net/save_net.ckpt")
        #         print("Accuracy: {0:.1%}".format(estimatedPrediction(train_set_x, One_hot_matrix_train)))

        saver.restore(sess, "my_net/save_net.ckpt")
        print("Accuracy: {0:.1%}".format( estimatedPrediction(train_set_x,One_hot_matrix_train) ))
        fun()

        # str = './TestImage/B/*.*'
        # Testing(str)
        # print(str)
        # str = './TestImage/P/*.*'
        # Testing(str)
        # print(str)
        # str = './TestImage/N/*.*'
        # Testing(str)
        # print(str)
        # str = './TestImage/O/*.*'
        # Testing(str)

################################################################################################

X=load.XX
y=load.yy
image_list = load.Image_list
train_set_x_orig = np.array(X)
train_set_y = np.array(y)
rem.trainImage(image_list)
m_train = len(train_set_y)
mask = 24604
train_set_x_flatten = featuresFunction(mask,image_list,0)

for i in range(train_set_x_flatten.shape[0]):
    for j in range(train_set_x_flatten.shape[1]):
        normalizeValue = max(normalizeValue, train_set_x_flatten[i][j])

train_set_x = train_set_x_flatten / normalizeValue

##############
num_classes = 4
One_hot_matrix_train = np.zeros(shape=(len(train_set_y), num_classes))
for i in range(len(train_set_y)):
    One_hot_matrix_train[i][train_set_y[i]] = 1

print(train_set_x.shape)
train_neural_network(x_var,train_set_x,One_hot_matrix_train)



