import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import random

import DataProcessing as load
import main as rem
from tensorflow.examples.tutorials.mnist import input_data
from xlwt import Workbook


n = len(load.XX)
arr = np.zeros(n)
for i in range (n):
    arr[i] = i

Number_of_features = 0
flag = 1
normalizeValue = 0.0

def featuresFunction(mask,image_list,image_list1):
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
        rem.GLCM(Train_Matrix,i,mask,0)#training

    Test_Matrix = np.zeros(shape=(Number_of_features,len(image_list1)))
    for i in range(len(image_list1)):
        rem.GLCM(Test_Matrix, i,mask,1)#testing
    global  flag
    if (flag == 0):
        wb = Workbook()
        sheet1 = wb.add_sheet('Train Matrix')
        sheet2 = wb.add_sheet('Test Matrix')
        for i in range(Train_Matrix.shape[0]):
            for j in range(Train_Matrix.shape[1]):
                sheet1.write(i,j,Train_Matrix[i][j])
        for i in range(Test_Matrix.shape[0]):
            for j in range(Test_Matrix.shape[1]):
                sheet2.write(i,j,Test_Matrix[i][j])
        wb.save('xlwt example.xls')
        #np.savetxt('Training_Data.txt', Train_Matrix, fmt='%2.2e', newline='\n\n')
        #np.savetxt('Testing_Data.txt', Test_Matrix, fmt='%2.2e', newline='\n\n')
    flag = 1

    return Train_Matrix,Test_Matrix


def featuresFunction1(mask,image_list,flagvalue):
    cnt = 0
    for i in range(15):
        if( (mask & (1<<i)) ):
            cnt = cnt + 1
    assert cnt>0
    global Number_of_features
    Number_of_features = cnt * 3
    Train_Matrix = np.zeros(shape=(Number_of_features,len(image_list)))
    for i in range(len(image_list)):
        rem.GLCM(Train_Matrix,i,mask,flagvalue)
    return Train_Matrix

#plt.imshow(load.XX[0])
#plt.show()

##############################################################################################
##### Neural Network Code
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
    test_set_x_flatten = featuresFunction1(mask, image_list1,1)
    test_set_x = test_set_x_flatten / normalizeValue
    ##############
    num_classes = 4
    One_hot_matrix_test = np.zeros(shape=(len(test_set_y), num_classes))
    for i in range(len(test_set_y)):
        One_hot_matrix_test[i][test_set_y[i]] = 1
    return estimatedPrediction(test_set_x,One_hot_matrix_test)>0.0

def train_neural_network(x_var,train_set_x,One_hot_matrix_train,test_set_x,One_hot_matrix_test,image_list1,y1): #,train_set_x,One_hot_matrix_train,test_set_x,One_hot_matrix_test
    # prediction = neural_network_model(x_var)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_var))  # v1.0 changes
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    epochs_no = 5001
    epoch_x = train_set_x.transpose()
    epoch_y = One_hot_matrix_train

    saver = tf.train.Saver()

    rem = 0.0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # v1.0 changes

        # training
        for epoch in range(epochs_no):
            _, c = sess.run([optimizer, cost], feed_dict={x_var: epoch_x, y_var: epoch_y})
            epoch_loss = c
            if(epoch%5000==0):
                print('Epoch', epoch, 'completed out of', epochs_no, 'loss:', epoch_loss)
                correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_var, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                acc = accuracy.eval({x_var: test_set_x.transpose(), y_var: One_hot_matrix_test})
                print('Accuracy:', acc)
                if(acc>rem):
                    rem = acc
                    saver.save(sess, "my_net1/save_net.ckpt")


        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_var, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        acc = accuracy.eval({x_var: test_set_x.transpose(), y_var: One_hot_matrix_test})
        print('Accuracy rate:', acc)
        saver.restore(sess, "my_net1/save_net.ckpt")
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_var, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        # print(accuracy.eval())
        acc = accuracy.eval({x_var: test_set_x.transpose(), y_var: One_hot_matrix_test})
        print('Accuracy rate:', acc )
        # save = np.zeros(shape=(4, 4))
        # for i in range(len(image_list1)):
        #     if(classChecking(image_list1[i], 0 )):
        #         save[(int)(y1[i])][0]+=1
        #     elif(classChecking(image_list1[i], 1 )):
        #         save[(int)(y1[i])][1]+=1
        #     elif (classChecking(image_list1[i], 2)):
        #         save[(int)(y1[i])][2]+=1
        #     else:
        #         save[(int)(y1[i])][3]+=1
        #
        # print(save)
        # Row = np.zeros(shape=(1, 4))
        # Col = np.zeros(shape=(4, 1))
        # precision = 0
        # recall = 0
        # for i in range(4):
        #     for j in range(4):
        #         Row[0][j] += save[i][j]
        #         Col[i][0] += save[i][j]
        #         if(j==3):
        #             recall += save[i][i] / Col[i][0]
        #         if(i==3):
        #             precision += save[j][j] / Row[0][j]
        #
        # precision /= 4.0
        # recall /= 4.0
        # print(precision)
        # print(recall)
        return acc
####################################################################################################

variable = x_var
total = 0.0
value = 5
runn = 3


for aa in range ( int(runn) ):
    random.shuffle(arr)
    q = int(n/value)
    p = -q
    for b in range( int(value) ):
        p = p + q
        image_list = []
        image_list1 = []
        X = []
        yy = []
        X1 = []
        y1 = []
        for k in range(n):
            if( k>=p and k<p+q ):
                X1.append(load.XX[int(arr[k])]) #image
                y1.append(load.yy[int(arr[k])]) #label
                image_list1.append(load.Image_list[int(arr[k])]) #directory
            else:
                X.append(load.XX[int(arr[k])])
                yy.append(load.yy[int(arr[k])])
                image_list.append(load.Image_list[int(arr[k])])

        train_set_x_orig = np.array(X)
        train_set_y = np.array(yy)
        test_set_x_orig = np.array(X1)
        test_set_y = np.array(y1)
        print(test_set_y)

        rem.trainImage(image_list)
        rem.testImage(image_list1)

        m_train = len(train_set_y)
        m_test = len(test_set_y)
        num_px = 64

        print(len(train_set_y))
        print(len(test_set_y))

        mask = 24604
        # mask = 32767
        print(mask) #0110 0000 0001 110

        train_set_x_flatten, test_set_x_flatten = featuresFunction(mask,image_list,image_list1)

        normalizeValue = 0.0
        for i in range(train_set_x_flatten.shape[0]):
            for j in range(train_set_x_flatten.shape[1]):
                normalizeValue = max(normalizeValue, train_set_x_flatten[i][j])
        for i in range(test_set_x_flatten.shape[0]):
            for j in range(test_set_x_flatten.shape[1]):
                normalizeValue = max(normalizeValue, test_set_x_flatten[i][j])
        # print(val)
        train_set_x_flatten = train_set_x_flatten / normalizeValue
        test_set_x_flatten = test_set_x_flatten / normalizeValue

        train_set_x = train_set_x_flatten
        test_set_x = test_set_x_flatten


        ##############
        num_classes = 4
        One_hot_matrix_train = np.zeros(shape=(len(train_set_y), num_classes))
        One_hot_matrix_test = np.zeros(shape=(len(test_set_y), num_classes))
        # print(One_hot_matrix_train.shape)
        for i in range(len(train_set_y)):
            One_hot_matrix_train[i][train_set_y[i]] = 1
        for i in range(len(test_set_y)):
            One_hot_matrix_test[i][test_set_y[i]] = 1
        ##############

        img_size_flat = Number_of_features

        total += train_neural_network(variable,train_set_x,One_hot_matrix_train,test_set_x,One_hot_matrix_test,image_list1,y1)


total = total / (runn*value)
print("Overall Accuracy: {0:.1%}".format(total))