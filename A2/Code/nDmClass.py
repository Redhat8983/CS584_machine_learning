'''
@author: Yung Chi Liu
'''
import matplotlib.pyplot as plt
import numpy as np
from pylab import *

class nDmC():
    def __init__(self):
        self.allDataX = np.array([])
        self.allDataY = np.array([]) 
        
        self.trainingX = np.array([])
        self.testingX = np.array([])
        self.trainingY = np.array([])
        self.testingY = np.array([])
        
        self.allDataSize = 0
        self.featureNum = 0
        
        self.InitReadData()
        
        self.classPool = []
        self.sizeOfPool = 0
        
        
    # Main program()
    def RunBody(self):
        
        foldNum = 5
        self.featureNum = len( self.allDataX)/ self.allDataSize  
        
        self.classPool.sort() 
        print self.classPool
        
        for i in range( foldNum):
            self.N_foldData(i, foldNum)
            
            self.trainingX.shape = -1, self.featureNum
            self.testingX.shape = -1, self.featureNum
            # sum the column
            #sumXais = np.sum( self.trainingX, axis = 0)
            
            # ====== calculate means ========
            meanPool = np.array([])
            for i in range( self.sizeOfPool):
                meanPool = append( meanPool, self.GetTrainingMean( self.classPool[i]) )
                
            meanPool.shape = -1, self.featureNum
            meanPool.tolist()
#             print "meanPool::", meanPool
            # ====== calculate means ========   
#             mean_zero = self.GetTrainingMean(0)
#             mean_1st = self.GetTrainingMean(1)
            
            # ====== calculate sigma ========
            sigmaPool = range( self.sizeOfPool)
            for i in range(0, self.sizeOfPool):
                sigmaPool[i] = np.matrix( np.zeros( (self.featureNum, self.featureNum)))
                sigmaPool[i] = self.GetTrainingSigma( self.classPool[i])
#             print "sigmaPool::", sigmaPool
            # ====== calculate sigma ========
#             sigma_zero = self.GetTrainingSigma(0)
#             sigma_1st = self.GetTrainingSigma(1)

            # ====== calculate alpha ========
            alphaPool = range( self.sizeOfPool)
            for i in range ( self.sizeOfPool):
                alphaPool[i] = 0.0
                alphaPool[i] = self.GetTrainingAlpha( self.classPool[i])
            # ====== calculate alpha ========
#             alpha_zero = self.GetTrainingAlpha(0)
#             alpha_1st = self.GetTrainingAlpha(1)
            
            
            
#             print "Testing", self.testingX
                         
            
            # define confusion matrix
            cft_matrix = np.array( np.zeros( (self.sizeOfPool, self.sizeOfPool)))
            
            for item in range( len( self.testingY)):
                
                # define discriminate function 
                # get g(x), put in list( disFuc)
                disFuc = range( self.sizeOfPool)
                for disF_i in range( self.sizeOfPool):
                    disFuc[ disF_i] = self.GetMemberFucTesting(item, meanPool[disF_i], sigmaPool[disF_i], alphaPool[disF_i])               

                    
                    
                # get argMax g(x)
                max_idx = disFuc.index( max( disFuc) )
                y_hat = self.classPool[max_idx]
                
#                 print "y_hat =>", y_hat, self.testingY[item], "<== Real"
                
                for col in range( self.sizeOfPool):
                    if int( self.testingY[item]) == int( self.classPool[col]):
                        cft_matrix[y_hat][col] += 1
                    else:
                        "Error in build confusion matrix!!" 
                 
            

            print "confusion matrix ::"        
            print cft_matrix;
            
            # measure error
            
            for baseOnI in range( self.sizeOfPool):
                print " Measure error base on class ", self.classPool[baseOnI]
                TP = cft_matrix[baseOnI][baseOnI]
                precision = float( TP/ ( np.sum( cft_matrix,axis =1))[baseOnI] )
                recall    = float( TP/ ( np.sum( cft_matrix,axis =0))[baseOnI] )
                accuracy = float( (np.sum( diag( cft_matrix)))/ len( self.testingY))
                F_measure = float( 2*precision*recall/ (precision+recall))
                
                print "Precision = ", precision
                print "   Recall = ", recall
                print " Accuracy = ", accuracy
                print "F-measure = ", F_measure
            
    def GetMemberFucTesting(self, itemNum, mean, sigma, alpha):
        
        # a_0 = |sigma|
        a_0 = np.linalg.det(sigma)
        a = float( - math.log( a_0)/2 )
        # b_0 = x(i)- mean
        b_0 = np.matrix( self.testingX[itemNum] - mean)
        
        b = float( -( np.dot( np.dot( b_0, np.linalg.inv(sigma)), b_0.T) )/2 )
        c = float( math.log( alpha))
        
        return a+b+c
        
    
        
        
        
    def GetTrainingMean(self,indicator): 
           
        total = np.zeros(self.featureNum)
        num_item = 0.0
        
        for i in range( len( self.trainingY)):
            
            if self.IndicateFuc(i, indicator):
                total += self.trainingX[i] 
                num_item += 1.0
        
        return total / num_item    
    
    def GetTrainingSigma(self, indicator):
        num_item = 0.0
        # beta = (x-mean)
        beta = np.zeros(self.featureNum)
        sigma = np.zeros( (self.featureNum, self.featureNum) )
        
        for i in range( len( self.trainingY)):
            
            if self.IndicateFuc(i, indicator):
                beta = np.matrix( self.trainingX[i] - self.GetTrainingMean(indicator) )                
                sigma += np.dot( beta.T, beta)
                num_item += 1.0
        
        return sigma / num_item   
    
    def GetTrainingAlpha(self, classNum):
    # Return a probability of class(classNum) in the training Data set.   
        itemNum = 0
        
        for i in range( len( self.trainingY)):
            if self.IndicateFuc( i, classNum ):
                itemNum += 1
                
        alpha = float( itemNum) / float( len( self.trainingY))
        
        return alpha
    
    def BoolInClass(self, value):
        for i in range( len( self.classPool)):
            if self.classPool[i] == int(value):
                return True
    
        return False
          
    
    # Load Data to :: allDataX, allDataY
    # Get value    :: allDataSize
    def InitReadData(self):
        switch = 0
        
        while ( switch == 0):
            option = raw_input ("Enter the data:( 1= 3class_iris, 2= pima-indians-diabetes.dat, 0 = exit program) ::")
            
            self.allDataX = ([])
            self.allDataY = ([]) 
        
            self.trainingX = ([])
            self.testingX = ([])
            self.trainingY = ([])
            self.testingY = ([])
            
            self.classPool = []
            self.meanPool=[]
            self.sigmaPool=[]
            self.alphaPool=[]
            self.sizeOfPool= 0
            
            if ( int(option) == 1):  #start to read data in np.array
                f = open('3class_iris.dat', 'r')
                
                for line in f.readlines():
                    #split the string on whitespace, return a list of numbers 
                    # (as strings)
                    numbers_str = line.replace('?', '-1').split("," )
                    #convert numbers to floats
                    numbers_float = [float(x) for x in numbers_str]  #map(float,numbers_str) works too
                    
                    self.allDataX = np.append( self.allDataX, numbers_float[: -1])
                    self.allDataY = np.append( self.allDataY, numbers_float[-1])
                    # store in class pool
                    if ( not self.BoolInClass( numbers_float[-1])):
                        self.classPool.append( int( numbers_float[-1]) )
                        
                    self.allDataSize = len( self.allDataY)
                    
                print self.allDataSize
                self.sizeOfPool = len( self.classPool)
#                 print self.classPool
#                 print "class in size::", len( self.classPool)
                f.close()
                
                self.RunBody()
            elif ( int(option) == 2):  #start to read data in np.array
                f = open('pima-indians-diabetes.dat', 'r')
                
                for line in f.readlines():
                    #split the string on whitespace, return a list of numbers 
                    # (as strings)
                    numbers_str = line.replace('?', '-1').split("," )
                    #convert numbers to floats
                    numbers_float = [float(x) for x in numbers_str]  #map(float,numbers_str) works too
                    
                    
                    self.allDataX = np.append( self.allDataX, numbers_float[: -1])
                    self.allDataY = np.append( self.allDataY, numbers_float[-1])
                    
                    # store in class pool
                    if ( not self.BoolInClass( numbers_float[-1])):
                        self.classPool.append( int( numbers_float[-1]) )
                    self.allDataSize = len( self.allDataY)
                    
                    
                print self.allDataSize
                self.sizeOfPool = len( self.classPool)
                f.close()
                
                self.RunBody()
            elif ( int(option) == 0 ):
                print ( "Exit the program...")
                switch = 1
            else: 
                print (" You entered a invalid data set::( please try again)...")
                           
    # set data to training and testing base on N fold validation
    # foldNum is round of data set at which part in the main data set.
    # total_foldNum is the N fold provide by program
    def N_foldData(self, foldNum, total_foldNum):
        
        dataRange = self.allDataSize / total_foldNum
        fetureNum = len( self.allDataX) / self.allDataSize
        print "------------------------------------------"
        print "Round ", foldNum, " ", total_foldNum, " Fold, processing..."
        # Init. np.array...
        self.trainingX = ([])
        self.trainingY = ([])  
        self.testingX = ([])
        self.testingY = ([])    
        
        self.trainingX = np.concatenate( ( self.allDataX[0:dataRange*fetureNum*foldNum], 
                                           self.allDataX[dataRange*fetureNum*(foldNum+1):]) )
        self.trainingY = np.concatenate( ( self.allDataY[0:dataRange*foldNum], 
                                           self.allDataY[dataRange*(foldNum+1):]) )        
        
        self.testingX = self.allDataX[ dataRange*fetureNum*foldNum : dataRange*fetureNum*(foldNum+1)]
        self.testingY = self.allDataY[ dataRange*foldNum : dataRange*(foldNum+1)]  
        
    # Give an indicator, if data[item] == indicator return 
    # ...     
    def IndicateFuc(self, item_num, indicator ):
        if int( self.trainingY[item_num] ) == indicator: 
            return True
        else:
            return False
                
        
if __name__ == "__main__":
    nDmC()
