from sklearn.datasets import fetch_mldata
import numpy as np
import os
from pylab import *
import random
from ctypes.wintypes import SIZE
import cvxopt
from cvxopt import solvers
import matplotlib.pyplot as plt
from sklearn import svm

dim = 2



def Read_data():
    option = raw_input ("Choose the data:( 1= sep_svm.data, 2= no_linear_sep.data, 3= circle.data::")
    if int(option) == 1:
        f = open('sep_svm.data', 'r')
    if int(option) == 2:
        f = open('no_linear_sep.data', 'r')
    if int(option) == 3:
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
#=========================================#
class soft_Poly_kernel_SVM_train(object):
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
        global dim
        num_item, num_features = X.shape
        K = np.zeros((num_item, num_item))        
    
        K =  (1 + np.dot(X, X.T)) ** dim
                
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
def hard_SVM_classify(X,y, weight, weight_zero):
    num_item, num_feature = X.shape
    
    for i in range( num_item):
        class_test = np.dot( weight, X[i]) + weight_zero
        print "Origin:", y[i], "test:", class_test
        
def Plot_line(X,y):
    # site from "http://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html"
    # figure number
    fignum = 1

    # fit the model
    for kernel in ('linear', 'poly', 'rbf'):
        clf = svm.SVC(kernel=kernel, gamma=2)
        clf.fit(X, y)
    
        # plot the line, the points, and the nearest vectors to the plane
        plt.figure(fignum, figsize=(8, 8))
        plt.clf()

    
        plt.axis('tight')
        x_min = 0
        x_max = 10
        y_min = 0
        y_max = 10
    
        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
    
        # Put the result into a color plot
        Z = Z.reshape(XX.shape)
        plt.figure(fignum, figsize=(4, 3))
        plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                    levels=[-.5, 0, .5])
    
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
    
        plt.xticks(())
        plt.yticks(())
        fignum = fignum + 1
        # =======================================
        num_item, num_feature = X.shape
    
        for i in range( num_item):
            if y[i] == 1:
                plt.plot( X[i][0], X[i][1], 'or')
            if y[i] == -1:
                plt.plot( X[i][0], X[i][1], 'ob')
        
    plt.show()    
    
       
if __name__ == "__main__": 
    X, y = Read_data()
    
    from sklearn.cross_validation import train_test_split
    
    Plot_line(X,y)
       
    


