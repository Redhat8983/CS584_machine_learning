
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
    
    
def Get_theta( x, y, size, polyNum):
    xlen = len(x)
    x.shape = xlen, -1
    
    temp_x = x
    for i in range(polyNum+1):
        poly_array = temp_x**i
        if i == 0:
            x = np.concatenate( (poly_array, x), axis = 1)
        elif i > 1:
            x = np.concatenate( (x, poly_array), axis = 1)
        
    bigZ = np.dot( inv(np.dot(np.transpose(x), x)), np.transpose(x))
    
    theta = np.dot( bigZ, y)

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
    print "Data set size is ", size_data
    size_test = input('Input test data size:')
    size_train = size_data - size_test
    # Set polynomial number
    polyNum =input( "Input polynomial for theta:")
    
    x_train = x[:-size_test]
    y_train = y[:-size_test]
    
    x_test = x[-size_test:]
    y_test = y[-size_test:]
    
    # calculate theta
    print "size train:", size_train
    theta = Get_theta( x_train, y_train, size_train, polyNum)
    print "Theta :", theta
     
 
    # calculate Y_hat
    Y_hat = np.array([])
    x_test_matrix = np.array([])

    for i in range(size_test):
        for power in range( polyNum+1):
            x_test_matrix = np.append( x_test_matrix, np.array( x_test[i]**power) )
            
        x_test_matrix.shape = 1,-1
        Y_hat= np.append( Y_hat, dot(x_test_matrix, theta)) 
        
        # initial x_test_matrix
        x_test_matrix = ([])
        
    # Y_hat_train
    Y_hat_train = np.array([])
    
    for i in range( size_train):
        for power in range( polyNum+1):
            x_test_matrix = np.append( x_test_matrix, np.array( x_train[i]**power) ) 
         
        x_test_matrix.shape = 1,-1
        Y_hat_train = np.append( Y_hat_train, dot( x_test_matrix, theta))
        
        x_test_matrix = ([])
        
    print "Training Error::", Get_error( y_train, Y_hat_train )
    print "Testing  Error::", Get_error( y_test, Y_hat)
   
    # sort x_test with Y_hat
    zip_list = zip( x_test, Y_hat)
    sort_list = sorted( zip_list, key = lambda x:x[0])
    
    x_test_sorted = [ x for x,y in sort_list]
    Y_hat_sorted = [ y for x,y in sort_list]
    
    # plt      
    plt.scatter(x_test, y_test,  color='black')
    plt.plot( x_test_sorted, Y_hat_sorted, color='blue',  linewidth=3)
  
    plt.xticks(())
    plt.yticks(())
  
    plt.show()
     
     
