'''
@author: YungChi Liu
'''
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
from sklearn.preprocessing import PolynomialFeatures


x = np.array([])
y = np.array([])

# Read data set, separate x and y data
def ReadFile():
    f = open('mvar-set1.dat', 'r')
    
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
      
def Get_theta( x, y, size, degree ):
    # get size for data
    xlen = len(y)
    x.shape = xlen, -1
    
    #x = np.insert( x, 0, 1, axis = 1)
       
    
    #degree
    poly = PolynomialFeatures(degree+1) 
    x = poly.fit_transform(x)  
    
    #bigZ = np.dot( inv(np.dot(np.transpose(x), x)), np.transpose(x))  
    #self.Theta = np.dot(pinv(Z),self.Training_Y_dataSet)
    theta = np.dot( pinv(x), y)

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
    
    out = 1;
    
    while out != 0:

        #get train and test range
        size_data = len(y)    
        #size_test = input('Input test data size:')
        size_test = size_data/10 #set manuly
        size_train = size_data - size_test
        # Set polynomial number
        #polyNum =input( "Input polynomial for theta:")
        feature_num = len(x) / size_data   # get number of feature in each line
        print "Data Size", size_data
        print "Feature_num::", feature_num
        
        combine = input("How many time of combination::")
        round_end = input("Run how many time::") 
        
        total_train_error = 0.0
        total_test_error = 0.0
        
        for round in range( round_end):
            
            print "Round::", round
        
            x_train = np.concatenate( ( x[0:size_test* feature_num*round], 
                                        x[size_test* feature_num*(round+1):]))
            y_train = np.concatenate( ( y[0:size_test*round], 
                                        y[size_test*(round+1):]))
            
            x_test = x[size_test* feature_num*round: size_test* feature_num*(round+1)]
            y_test = y[size_test*round:size_test*(round+1)]
        
        # calculate theta
            print "size train:", size_train
            theta = Get_theta( x_train, y_train, size_train, combine)
            #print "Theta :", theta
            
            
            
            # calculate Y_hat
            Y_hat = np.array([])
            x_test_matrix = np.array([1])
            print "GG", len(theta)
            #print theta
            theta.shape = feature_num+1, -1
        
            for i in range(size_test):
                for f in range( feature_num):
                    x_test_matrix = np.append( x_test_matrix, np.array( x_test[i*feature_num + f]) )   
                    
                x_test_matrix.shape = 1,-1
                Y_hat= np.append( Y_hat, dot(x_test_matrix, theta)) 
                
                # initial x_test_matrix
                x_test_matrix = ([1])
                   
            # Y_hat_train
            Y_hat_train = np.array([])
            x_train.shape = -1, 1
             
            for i in range( size_train):
                for f in range( feature_num):
                    x_test_matrix = np.append( x_test_matrix, np.array( x_train[i*feature_num + f]) )
                 
                x_test_matrix.shape = 1,-1
                Y_hat_train = np.append( Y_hat_train, dot( x_test_matrix, theta))
                
                x_test_matrix = ([1])
                
                
            temp_train_error = Get_error( y_train, Y_hat_train )
            temp_test_error = Get_error( y_test, Y_hat)
                
            print "Training Error::", temp_train_error
            print "Testing  Error::", temp_test_error
            
            total_train_error += temp_train_error
            total_test_error += temp_test_error
            
         
        print "=================================================================="   
        print "Combination time:", combine 
        print round_end, "Time avg. Training Error:", total_train_error/round_end    
        print round_end, "Time avg. Testing Error:", total_test_error/round_end  
        
        out = input("Quit enter 0, continue enter1::") 

