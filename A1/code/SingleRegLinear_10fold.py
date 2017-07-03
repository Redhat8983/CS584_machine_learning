
'''
@author: YungChi Liu
'''
import matplotlib.pyplot as plt
import numpy as np
from pylab import *


x = np.array([])
y = np.array([])

# Read data set, separate x and y data
def ReadFile():
    f = open('svar-set4.dat', 'r')
    
    global x
    global y

    for line in f.readlines():
        # read line
        
        if line[0] != '#':
            token = map( float, line.split() )
            x = np.append(x, token[:-1])
            y = np.append(y, token[-1] )
        # read line finish & ignored the '#' started line
        # separate data to x & y matrixes
        
    f.close()
def Get_theta( x, y, size):
    A = np.array( [0.0,0.0,0.0,0.0], dtype = np.float )
    B = np.array( [0.0,0.0], dtype = np.float )
    A[0] = size
    
    A.shape = 2,2
    B.shape = 2,1
    
    for i in range(0,size):
        A[0][1] += x[i]
        A[1][1] += x[i]**2
        B[0] += y[i]
        B[1] += x[i]*y[i]    
        A[1][0] = A[0][1]
        
     
    theta = solve(A,B)  
    #theta = np.linalg.solve(A,B)
    return theta

def Get_error( y, y_hat ):
    size = len(y)
    total = 0.0
    for i in range( size ):
        dif_sqe = ( y[i] - y_hat[i] )**2
        total += dif_sqe
    
    return total/ size

if __name__ == "__main__": 
    # read data to x[] and y[]
    ReadFile()

    #get train and test range
    size_data = len(x)
    print "Data set size is :", size_data
    
    # get test set size
    size_test = size_data/10
    size_train = size_data - size_test
    
    total_train_error = 0
    total_test_error = 0
    
    for round in range(10):
        
        x_train = np.concatenate( ( x[0:size_test*round], 
                                    x[size_test*(round+1):]))
        y_train = np.concatenate( ( y[0:size_test*round], 
                                    y[size_test*(round+1):]))
        
        x_test = x[size_test*round:size_test*(round+1)]
        y_test = y[size_test*round:size_test*(round+1)]
        
        # calculate theta
        theta = Get_theta( x_train, y_train, size_train)
        #print "Theta :", theta       
        
        Y_hat = np.array([])
        for i in range(0,size_test):
            X = np.array( [1, x_test[i]] )
            X.shape = 1,2
            Y_hat= np.append( Y_hat, dot(X, theta)) 
         
        Y_hat_train = np.array([])
        for i in range(0,size_train):
            X = np.array( [1, x_train[i]] )
            X.shape = 1,2
            Y_hat_train= np.append( Y_hat_train, dot(X, theta)) 
        
        total_train_error += Get_error( y_train, Y_hat_train)
        total_test_error += Get_error( y_test, Y_hat)
        
    
    total_train_error /= 10
    total_test_error /= 10
        
    print "With 10 fold cross validation"
    print "Training Error ::", total_train_error
    print "Testing Error  ::", total_test_error
