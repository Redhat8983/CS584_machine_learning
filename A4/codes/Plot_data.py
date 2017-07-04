from sklearn.datasets import fetch_mldata
import numpy as np
import os
from pylab import *
import pylab as plt
import random
from ctypes.wintypes import SIZE
import cvxopt
from cvxopt import solvers



def Read_data():
    f = open('circle.data', 'r')
    extract_x = np.array([])
    extract_y = np.array([])
                
    for line in f.readlines():
        numbers_str = line.split("," )
        #convert numbers to floats
        numbers_float = [float(x) for x in numbers_str]  #map(float,numbers_str) works too
                    
        extract_x = np.append( extract_x, numbers_float[1:])
        extract_y = np.append( extract_y, numbers_float[0])
        size = len( extract_y)
        f.close()
        
        extract_x.shape = -1,2
        extract_y.shape = -1,1
    
    return extract_x, extract_y
#=====================================#
def Plot_picture( X, y):
    num_item, num_feature = X.shape
    
    
    for i in range( num_item):
        if y[i] == 1:
            plt.plot( X[i][0], X[i][1], 'or')
        if y[i] == -1:
            plt.plot( X[i][0], X[i][1], 'ob')
    
    
    plt.xlabel('X')
    plt.ylabel('Y')
    
    
    plt.show()
    
       
if __name__ == "__main__": 
    X, y = Read_data()
    Plot_picture( X, y )
    
    
    
    

