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
    class_2_start_index = 0
    class_2_test_start_index = 0
    
    for i in range ( len(y_train)):
        if y_train[i] == 1:
            class_1_start_index = i
            break  
    for i in range( len(y_test)):
        if y_test[i] == 1:
            class_1_test_start_index = i  
            break
    for i in range( len(y_train)):
        if y_train[i] == 2:
            class_2_start_index = i
            break
    for i in range( len(y_test)):
        if y_test[i] == 2:
            class_2_test_start_index = i  
            break
    
            

    # extract X
    extract_x_train = np.concatenate( ( X_train[0:900],
                                        X_train[class_1_start_index:class_1_start_index+900]))
    extract_x_train = np.concatenate( ( extract_x_train,
                                        X_train[class_2_start_index:class_2_start_index+900]))
    extract_x_test  = np.concatenate( ( X_test[0:100], 
                                        X_test[class_1_test_start_index: class_1_test_start_index+100]))
    extract_x_test  = np.concatenate( ( extract_x_test,
                                        X_test[class_2_test_start_index: class_2_test_start_index+100]))
    extract_x = np.concatenate( ( extract_x_train, extract_x_test) )
    # extract y(class)
    extract_y_train = np.concatenate( ( y_train[0:900],
                                        y_train[class_1_start_index:class_1_start_index+900]))
    extract_y_train = np.concatenate( ( extract_y_train,
                                        y_train[class_2_start_index:class_2_start_index+900]))
    extract_y_test  = np.concatenate( ( y_test[0:100], 
                                        y_test[class_1_test_start_index: class_1_test_start_index+100]))
    extract_y_test  = np.concatenate( ( extract_y_test,
                                        y_test[class_2_test_start_index: class_2_test_start_index+100]))
    extract_y = np.concatenate( ( extract_y_train, extract_y_test) )
    # zip and shuffle extract_x, extract_y
#     combined = zip( extract_x, extract_y)
#     random.shuffle(combined)
#     extract_x[:], extract_y[:] = zip(*combined)
    
    return extract_x, extract_y
#================================================================#
def Extract_data_Balance():
    f = open('balance_3class.data', 'r')
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
'''
Indicator:
    Give a indicator, and a class j
    if j is class indicator return 1
    otherwise               return 0
'''
def Indicator( indicator, j ):
    if indicator == int(j):
        return 1
    else:
        return 0
def SoftMax( theta,x, indicate):
    den =0.0
    numerator = 0.0
    
    for i in range(3):
        den = den + math.exp( np.dot( theta[i], x.T) )
        
    numerator = math.exp( np.dot( theta[indicate], x.T) )
    
    softmax = numerator/den
    
    return softmax
    
def Theta_update( lr,theta, x, y, size,indicate):
    temp_sum = np.zeros( len(x[0]), float)
    
    for i in range(size):     
        a = SoftMax(theta, x[i], indicate)- Indicator(indicate, y[i]) 
        temp_sum = temp_sum + a*x[i]
     
    new_theta = theta[indicate] - lr*temp_sum
    
    return new_theta

def Get_max_softmax( theta, x):
    max_i = 0
    max = SoftMax(theta, x, 0)
    
    for i in range(3):  
        if SoftMax(theta, x, i) > max:
            max = SoftMax(theta, x, i)
            max_i = i 
            
    return max_i     
"""
main:
    do the main body here:
    1.build 10 fold validation data.(for)
    2.Run theta update function K times(for) // stop point ( error < 0.1 or GD f times)
    3.classify(for)
    4.calculate testing error.
""" 

def linear_logistic_regression( x, y): 
    from sklearn.cross_validation import KFold
    kf = KFold( len(x) , n_folds= 10)
    sum_test_error = 0.0
    sum_GD = 0.0
    
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
        #cft_matrix = np.array( np.zeros( (self.sizeOfPool, self.sizeOfPool)))
        theta = np.array( np.zeros( (3,element_size), float))
        theta[0] = np.random.rand(1,element_size)*0.1
        theta[1] = np.random.rand(1,element_size)*0.1
        theta[2] = np.random.rand(1,element_size)*0.1
        
        learning_rate = 0.0001
        train_error = 1.0
        GD_time = 0
        
        
        while GD_time < 1000 and train_error >= 0.1:
            theta[0] = Theta_update( learning_rate,theta, x_train, y_train,train_size, 0 )
            theta[1] = Theta_update( learning_rate,theta, x_train, y_train,train_size, 1 )
            theta[2] = Theta_update( learning_rate,theta, x_train, y_train,train_size, 2 )
                
            # compute training error
            train_error = 0.0
            for i in range(train_size):
                train_yhat = Get_max_softmax( theta, x_train[i])
                if y_train[i] == float(0):
                    if train_yhat != 0:
                        train_error += 1.0
                if y_train[i] == float(1):
                    if train_yhat != 1:
                        train_error += 1.0
                if y_train[i] == float(2):
                    if train_yhat != 2:
                        train_error += 1.0
                        
            train_error /= train_size
            
            if GD_time%50 == 0:
                print GD_time, " times, Train Error rate =", train_error 
                       
            GD_time += 1
        # End while ( GD_time >= 100 || train_error < 0.1)
        print "GD times:", GD_time, " training error", train_error
        sum_GD += GD_time
        
        # run test case
        test_error = 0.0
        for i in range( test_size):
            test_yhat = Get_max_softmax( theta, x_test[i])
            if y_test[i] == float(0):
                if test_yhat != 0:
                    test_error += 1.0
            if y_test[i] == float(1):
                if test_yhat != 1:
                    test_error += 1.0
            if y_test[i] == float(2):
                if test_yhat != 2:
                    test_error += 1.0
                    
            
        test_error /= test_size           
        print "test error rate::", test_error
        sum_test_error += test_error
        print "========================"
    test_error = sum_test_error/10
    print "10Fold, average testing acc:", 1.0-test_error
    print "Avg. GD:", sum_GD/10
    
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
    