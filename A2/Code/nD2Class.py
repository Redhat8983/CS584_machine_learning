'''
@author: Yung Chi Liu
'''
import matplotlib.pyplot as plt
import numpy as np
from pylab import *

class nD2C():
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
        
        
    # Main program()
    def RunBody(self):
        
        foldNum = 10
        self.featureNum = len( self.allDataX)/ self.allDataSize 
        
        totalTP = 0.0
        totalFN = 0.0
        totalTN = 0.0
        totalFP = 0.0
        
        precision =0.0
        recall =0.0
        accuracy =0.0
        F_maesure  = 0.0 
        
        for i in range( foldNum):
            self.N_foldData(i, foldNum)
            
            self.trainingX.shape = -1, self.featureNum
            self.testingX.shape = -1, self.featureNum
            # sum the column
            #sumXais = np.sum( self.trainingX, axis = 0)
            
            mean_zero = self.GetTrainingMean(0)
            mean_1st = self.GetTrainingMean(1)

            sigma_zero = self.GetTrainingSigma(0)
            sigma_1st = self.GetTrainingSigma(1)
            
            alpha_zero = self.GetTrainingAlpha(0)
            alpha_1st = self.GetTrainingAlpha(1)
            
            TP = 0.0
            FN = 0.0
            TN = 0.0
            FP = 0.0
            
#             print "Testing", self.testingX
                         
            # start discriminate function 
            for i in range( len( self.testingY)):
                g0 = self.GetMemberFucTesting(i, mean_zero, sigma_zero, alpha_zero)
                g1 = self.GetMemberFucTesting(i, mean_1st, sigma_1st, alpha_1st)
                
                if int(self.testingY[i]) == 0:
                    if (g0-g1)>= 0:
                        TP += 1
                    else:
                        FN += 1
                elif int(self.testingY[i]) == 1:
                    if (g1-g0)>= 0:
                        TN += 1
                    else:
                        FP += 1
                else:
                    print "This should not happened!!!."
                    
            print "   P   N"
            print "P ", TP, "  ", FP
            print "N ", FN, "  ", TN
            print " "
            
            totalTP += TP
            totalFP += FP
            totalFN += FN
            totalTN += TN
            
            precision += float(TP/ (TP+FP) )
            recall += float( TP/ (TP+FN))
            accuracy += float( (TP+TN)/ len( self.testingY))
            F_maesure += float( 2*precision*recall/ (precision+recall))
        #===================print result===========================================     
            
        totalTP /= foldNum
        totalFP /= foldNum
        totalFN /= foldNum
        totalTN /= foldNum 
        print "Testing Size::", self.allDataSize/foldNum
        
        print foldNum, " -validation"
        print "   P   N"
        print "P ", totalTP, "  ", totalFP
        print "N ", totalFN, "  ", totalTN
        print " "  
        
        print "Precision =", precision/foldNum
        print "Recall =", recall/foldNum
        print "Accuracy =", accuracy/foldNum
        print "F-maesure= ", F_maesure/foldNum
            
            
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
        
    
    # Load Data to :: allDataX, allDataY
    # Get value    :: allDataSize
    def InitReadData(self):
        switch = 0
        
        while ( switch == 0):
            option = raw_input ("Enter the data:( 1= 1_iris.dataa, 2=1_mm.data(with 2 feature), 0 = exit program) ::")
            
            self.allDataX = ([])
            self.allDataY = ([]) 
        
            self.trainingX = ([])
            self.testingX = ([])
            self.trainingY = ([])
            self.testingY = ([])
            
            if ( int(option) == 1):  #start to read data in np.array
                f = open('2_nD2C.data', 'r')
                
                for line in f.readlines():
                    #split the string on whitespace, return a list of numbers 
                    # (as strings)
                    numbers_str = line.replace('?', '-1').split("," )
                    #convert numbers to floats
                    numbers_float = [float(x) for x in numbers_str]  #map(float,numbers_str) works too
                    
                    self.allDataX = np.append( self.allDataX, numbers_float[: -1])
                    self.allDataY = np.append( self.allDataY, numbers_float[-1])
                    self.allDataSize = len( self.allDataY)
                    
                print self.allDataSize
                f.close()
                
                self.RunBody()
            elif ( int(option) == 2):  #start to read data in np.array
                f = open('1_mm.data', 'r')
                
                for line in f.readlines():
                    #split the string on whitespace, return a list of numbers 
                    # (as strings)
                    numbers_str = line.replace('?', '-1').split("," )
                    #convert numbers to floats
                    numbers_float = [float(x) for x in numbers_str]  #map(float,numbers_str) works too
                    
                    
                    self.allDataX = np.append( self.allDataX, numbers_float[:2])
                    self.allDataY = np.append( self.allDataY, numbers_float[-1])
                    self.allDataSize = len( self.allDataY)
                    
                print self.allDataSize
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
    nD2C()
