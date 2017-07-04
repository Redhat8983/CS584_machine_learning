from sklearn.datasets import fetch_mldata
import numpy as np
import os
from pylab import *
import random
from ctypes.wintypes import SIZE


'''
extract data from MNIST
    extract 900 items from testing, 100 items from testing with label(class) 0
    and label 1, with total 2000 items.
    shuffle the data.
Return data:: x_train(np.array), y_train(np.array)
'''
def Extract_data():
    mnist = fetch_mldata('MNIST original')
    mnist.data.shape
    mnist.target.shape
    np.unique(mnist.target)

    X = mnist.data / 255.
    y = mnist.target    
    
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
     
    class_1_start_index = 0
    class_1_test_start_index = 0
    
    for i in range ( len(y_train)):
        if y_train[i] == 1:
            class_1_start_index = i
            break  
    for i in range( len(y_test)):
        if y_test[i] == 1:
            class_1_test_start_index = i  
            break

    # extract X
    extract_x_train = np.concatenate( ( X_train[0:900],
                                        X_train[class_1_start_index:class_1_start_index+900]))
    extract_x_test  = np.concatenate( ( X_test[0:100], 
                                        X_test[class_1_test_start_index: class_1_test_start_index+100]))
    extract_x = np.concatenate( ( extract_x_train, extract_x_test) )
    # extract y(class)
    extract_y_train = np.concatenate( ( y_train[0:900],
                                        y_train[class_1_start_index:class_1_start_index+900]))
    extract_y_test  = np.concatenate( ( y_test[0:100], 
                                        y_test[class_1_test_start_index: class_1_test_start_index+100]))
    extract_y = np.concatenate( ( extract_y_train, extract_y_test) )
    # zip and shuffle extract_x, extract_y
#     combined = zip( extract_x, extract_y)
#     random.shuffle(combined)
#     extract_x[:], extract_y[:] = zip(*combined)
    
    return extract_x, extract_y
#================================================================#
def Extract_data_Balance():
    f = open('balances_2class.data', 'r')
    extract_x = np.array([])
    extract_y = np.array([])
                
    for line in f.readlines():
        featureNum = 4
        numbers_str = line.split("," )
        #convert numbers to floats
        numbers_float = [float(x) for x in numbers_str]  #map(float,numbers_str) works too
                    
        extract_x = np.append( extract_x, numbers_float[1:])
        extract_y = np.append( extract_y, numbers_float[0])
        size = len( extract_y)
        f.close()
        
        extract_x.shape = -1,4
    
    return extract_x, extract_y
#=========================================#
'''
Calculate: Sigmoid function(x)
    theta update fuction()

'''
def Sigmod( theta,x):
    # h(theta) = 1 / ( 1+ exp( - theta * x)
    # theta = [1*n], x = [n*1]. n: # of features.
    den = 1.0 + math.exp(-np.dot(theta, x.T))
    sigmoid = 1.0/den
    
    return sigmoid

def Theta_update( lr,theta, x, y, size):
    temp_sum = np.zeros( len(x[0]), float)
    
    for i in range(size):     
        a = Sigmod(theta, x[i])- y[i]
        temp_sum = temp_sum + a*x[i]
        
    new_theta = theta - lr*temp_sum

    return new_theta
        
"""
main:
    do the main body here:
    1.build 10 fold validation data.(for)
    2.Run theta update function K times(for) // stop point ( error < 0.1 or GD f times)
    3.classify(for)
    4.calculate testing error.
""" 

def linear_logistic_regression( x, y): 
    x = np.insert(x, 0, 1.0,  axis = 1)
    from sklearn.cross_validation import KFold
    kf = KFold( len(x) , n_folds= 10)
    sum_test_error = 0.0
    
    for train_index, test_index in kf:
        # allocate training and test data
        x_train = x[train_index]
        y_train = y[train_index]
        x_test = x[test_index]
        y_test = y[test_index]
        
        # extract size info.
        element_size = len(x_train[0])
        train_size = len(y_train)
        test_size = len(y_test)
        
        # set parameters
        theta = np.random.rand(1,element_size)*0.1
        learning_rate = 0.00001
        train_error = 1.0
        GD_time = 0
        
        while GD_time < 1000 and train_error >= 0.1:
            theta = Theta_update( learning_rate,theta, x_train, y_train, train_size )
            # compute training error
            train_error = 0.0
            for i in range(train_size):
                train_yhat = Sigmod(theta, x_train[i])
                if y_train[i] == float(0):
                    if train_yhat >= 0.5:
                        train_error += 1.0
                if y_train[i] == float(1):
                    if float(train_yhat) < 0.5:
                        train_error += 1.0
                        
            train_error /= train_size
            
            if GD_time%50 == 0:
                print GD_time, " times, Train Error rate =", train_error 
                       
            GD_time += 1
        # End while ( GD_time >= 100 || train_error < 0.1)
        print "GD times:", GD_time, " training error", train_error
        
        # run test case
        test_error = 0.0
        for i in range( test_size):
            test_yhat = Sigmod(theta, x_test[i])
            if y_test[i] == float(0):
                if test_yhat >= 0.5:
                    test_error += 1.0
            elif y_test[i] == float(1):
                if train_yhat < 0.5:
                    test_error += 1.0
                    
            
        test_error /= test_size           
        print "test error rate::", test_error
        sum_test_error += test_error
        print "========================"
    test_error = sum_test_error/10
    print "10Fold, average testing acc:", 1.0-test_error
    
def Init_interface():
    exit = False
    while( not exit):
        option = raw_input ("Choose the data:( 1= MNIST data, 2= Balance Data, 0 = exit program) ::")
        if ( int(option) == 1):
            x,y = Extract_data()
            linear_logistic_regression(x, y)
        elif ( int(option) == 2):
            x,y = Extract_data_Balance()
            linear_logistic_regression(x, y)
        elif ( int(option)== 0):
            exit = True
            print "Exiting the program"
    
    
if __name__ == "__main__": 
    Init_interface()
    