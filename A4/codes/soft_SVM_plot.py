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
    option = raw_input ("Choose the data:( 1= sep_svm.data, 2= no_linear_sep.data, 3= circle.data::")
    if int(option) == 1:
        f = open('sep_svm.data', 'r')
    if int(option) == 2:
        f = open('no_linear_sep.data', 'r')
    if int(option) == 3:
        f = open('circle.data', 'r')
    if int(option) == 4:
        f = open('unbalance.data', 'r')
        
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
#=========================================#
class soft_SVM_train(object):
    def __init__(self):
        print "_init_hard_SVM_train..."
        
    def Get_weights(self, X, y):
        
        lagrange_multi = self.Get_lagrange_multiplier(X, y)
        weight = self.Get_weight(lagrange_multi, X, y)
        weight_zero = self.Get_weight_zero(lagrange_multi, X, y, weight)
        
        print "W:",weight, "W0:", weight_zero
        
        
        return weight, weight_zero
    
    #=============================================#
    
    def Gram_matrix(self, X):
        num_item, num_features = X.shape
        K = np.zeros((num_item, num_item))        
        K = np.dot( X, X.T)
                
        return K
        
    def Get_lagrange_multiplier(self, X, y):
        print "lagrange_multi"
        num_item, num_features = X.shape
               
        # Solves
        # min 1/2 x^T P x + q^T x
        # s.t.
        #  Ax = b 
        #  Gx <= h
        XXT = self.Gram_matrix(X)
        P = cvxopt.matrix( np.dot(y, y.T) * XXT)
        q = cvxopt.matrix(-1 * np.ones(num_item).T)
        
        G_up = cvxopt.matrix(np.diag(np.ones(num_item).T * -1))
        G_down = cvxopt.matrix( np.diag( np.ones(num_item).T ))
        h_up = cvxopt.matrix(np.zeros(num_item).T)
        h_down = cvxopt.matrix(np.ones(num_item).T)
        
        G = cvxopt.matrix(np.vstack((G_up, G_down)))
        h = cvxopt.matrix(np.vstack((h_up, h_down)))
        
        A = cvxopt.matrix(y, (1, num_item))
        b = cvxopt.matrix(0.0)
        
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        # get lagrange multiplier      
        alpha = np.asarray( np.ravel( solution['x']) )
        
                
        return alpha
    
    #============================================#
    def Get_weight(self, lagrange, X, y):
        num_item, num_feature = X.shape
        
        weight = np.zeros(( 1, num_feature))
        
        for i in range(num_item):
            local_weight = lagrange[i]*y[i]*X[i]
            weight += local_weight
    
        return weight    
    
    def Get_weight_zero(self, lagrange, X, y, weight):
        num_item, num_feature = X.shape
        
        num_support_vector = 0.0
        sigma = 0.0
        
        for i in range (num_item):
            if ( lagrange[i] > 0.0 ):
                num_support_vector+= 1
                
                local_sum = y[i]- np.dot( weight, X[i])
                sigma += local_sum
        
        weight_zero = (1/num_support_vector)*sigma
        
        return weight_zero
    
#=====================================#
def soft_SVM_classify(X,y, weight, weight_zero):
    num_item, num_feature = X.shape
    
    error_sample = 0.0
    
    for i in range( num_item):
        class_test = np.dot( weight, X[i]) + weight_zero
        print "Origin:", y[i], "test:", class_test
        if y[i] == 1 and class_test < 0 :
            error_sample += 1
        if y[i] == -1 and class_test > 0 :
            error_sample += 1
            
            
    acc = 1 -error_sample/num_item
    print "----->", acc
    return acc
#=====================================#
def Plot_picture( X, y, weight, w0):
    num_item, num_feature = X.shape
    
    # compute 2 point for boundary
    boundary_X = np.zeros( (1,2))
    boundary_Y = np.zeros( (1,2))
    boundary_X[0][0] = max(X[:,0])
    boundary_X[0][1] = min(X[:,0])
    
    # W.T*X + W0 = 0
    # X2 = ( -W0 - W1[0][0]*X1)/ W2[0][1]
    boundary_Y[0][0] = ( -1*w0 - weight[0][0]* boundary_X[0][0])/ weight[0][1]
    boundary_Y[0][1] = ( -1*w0 - weight[0][0]* boundary_X[0][1])/ weight[0][1]
    boundary_X.shape = 2, 1
    boundary_Y.shape = 2, 1
    
    # compute W.TX+W0 = +1, -1
    boundary_Y_Plus = np.zeros( (2,1))
    boundary_Y_Minus = np.zeros((2,1))
    
    boundary_Y_Plus[0] = ( -1*w0 - weight[0][0]* boundary_X[0] +1) / weight[0][1]
    boundary_Y_Plus[1] = ( -1*w0 - weight[0][0]* boundary_X[1] +1) / weight[0][1]
    
    boundary_Y_Minus[0] = ( -1*w0 - weight[0][0]* boundary_X[0]-1) / weight[0][1]
    boundary_Y_Minus[1] = ( -1*w0 - weight[0][0]* boundary_X[1]-1) / weight[0][1]
    
    for i in range( num_item):
        if y[i] == 1:
            plt.plot( X[i][0], X[i][1], 'or')
        if y[i] == -1:
            plt.plot( X[i][0], X[i][1], 'ob')
    
    
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.plot(boundary_X, boundary_Y, 'k')
    plt.plot(boundary_X, boundary_Y_Plus, 'k--')
    plt.plot(boundary_X, boundary_Y_Minus, 'k--')
    plt.show()
    
       
if __name__ == "__main__": 
    X, y = Read_data()
    
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
    
    weight, weight_zero = soft_SVM_train().Get_weights( X_train, y_train)
    
    acc = soft_SVM_classify(X_test, y_test, weight, weight_zero)
    
    Plot_picture( X, y, weight, weight_zero)
    
    # 10 fold validation
    from sklearn.cross_validation import KFold
    kf = KFold( len(X) , n_folds= 10,shuffle=True, random_state=42)
    sum_test_acc = 0.0
    
    for train_index, test_index in kf:
        # allocate training and test data
        X_trainf = X[train_index]
        y_trainf = y[train_index]
        X_testf = X[test_index]
        y_testf = y[test_index]
        
        weightf, weight_zerof = soft_SVM_train().Get_weights( X_trainf, y_trainf)
        local_acc = soft_SVM_classify(X_testf, y_testf, weightf, weight_zerof)
        sum_test_acc += local_acc
        
    print "10 Fold Validation:", sum_test_acc/10
    
    
