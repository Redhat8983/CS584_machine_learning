'''
@author: Yung Chi Liu

Main goals:
  1. Read Data, Store Data, 
  2. Classified Data by "class"
  3. Set "training" and "testing" sets
  
  4. try to store in np.matrix.
'''

'''
  Data Sets:
  1. 1D2Class Data "1D2C_mm.data": Only use feature 1 and 2, 3 4 5 had missing data.
     in 1D2Class case, I'll choose feature 2 (age) as use able feature.
''' 

'''
Idea:
  Reader(){
    Self.Init // choose data
    
    Self.nD2C // Read data to structure
    {
      Store Local to Self.
    } // nD2C
    
    Self.Sep_data // Set data to training and testing sets
                  // Do classification
                  // for 1D2class suppose to have class_1_traing, 
                  // class_2_traing, class_1_testing, class_2_tesing ( 4 arrays)
  } // reader
'''
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
import math

class _1D2C():
    def __init__(self):
        self.allDataX = np.array([])
        self.allDataY = np.array([]) 
        
        self.trainingX = np.array([])
        self.testingX = np.array([])
        self.trainingY = np.array([])
        self.testingY = np.array([])
        
        self.allDataSize = 0
        
        self.InitReadData()
        
        
    # Main program()
    def RunBody(self):
        
        foldNum = 10
        
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
            
            mean_Zero = self.GetTrainingMean(0.0)
            mean_1st = self.GetTrainingMean(1.0)
            
            sigma_Zero = self.GetTrainingSigma(0)
            sigma_1st = self.GetTrainingSigma(1)
            
            print "mean 0 = ", mean_Zero, ", mean 1=", mean_1st
            print "sigma 0 = ", sigma_Zero, ", sigma 1= ", sigma_1st
            
            alpha_Zero = self.GetTrainingAlpha(0)
            alpha_1st = self.GetTrainingAlpha(1)
            
            print "alpha 0 = ", alpha_Zero, ", alpha 1 = ", alpha_1st
            
            # Run training set
            # No need for now.
                
            # Discriminate Function 
            # use d(x) = g0(x) - g1(x)
           
            TP = 0.0
            FN = 0.0
            TN = 0.0
            FP = 0.0
            
            print "total item # is ", len(self.testingY)
        
            
            
            for i in range( len( self.testingY)):
                g0 = self.GetMemberFucTesting(i, 0, mean_Zero, sigma_Zero, alpha_Zero)
                g1 = self.GetMemberFucTesting(i, 1, mean_1st, sigma_1st, alpha_1st)
                 
                 
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
    
    # Load Data to :: allDataX, allDataY
    # Get value    :: allDataSize
    def InitReadData(self):
        switch = 0
        
        while ( switch == 0):
            print "********************************"
            option = raw_input ("Enter the data:( 1= 1_mm.data, 2= 1_iris.data = exit program) ::")
            
            self.allDataX = ([])
            self.allDataY = ([]) 
        
            self.trainingX = ([])
            self.testingX = ([])
            self.trainingY = ([])
            self.testingY = ([])
            
            if ( int(option) == 1):  #start to read data in np.array
                f = open('1_mm.data', 'r')
                
                for line in f.readlines():
                    featureNum = 1
                    #token = map( int, line.replace('?', '-1').split("," ) )
                    
                    #split the string on whitespace, return a list of numbers 
                    # (as strings)
                    numbers_str = line.replace('?', '-1').split("," )
                    #convert numbers to floats
                    numbers_float = [float(x) for x in numbers_str]  #map(float,numbers_str) works too
                    
                    
                    self.allDataX = np.append( self.allDataX, numbers_float[1])
                    self.allDataY = np.append( self.allDataY, numbers_float[-1])
                    self.allDataSize = len( self.allDataY)
                    
                print self.allDataSize
                f.close()
                
                self.RunBody()
            elif ( int(option) == 2 ):
                f = open('1_iris.data', 'r')
                
                for line in f.readlines():
                    featureNum = 1
                    #token = map( int, line.replace('?', '-1').split("," ) )
                    
                    #split the string on whitespace, return a list of numbers 
                    # (as strings)
                    numbers_str = line.replace('?', '-1').split("," )
                    #convert numbers to floats
                    numbers_float = [float(x) for x in numbers_str]  #map(float,numbers_str) works too
                    
                    
                    self.allDataX = np.append( self.allDataX, numbers_float[0])
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
        print "==== ==== ==== ==== ==== ==== ===="
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
        if self.trainingY[item_num] == indicator: 
            return True
        else:
            return False
        
    def GetTrainingMean(self,indicator): 
           
        total = 0.0
        num_item = 0.0
        
        for i in range( len( self.trainingY)):
            
            if self.IndicateFuc(i, indicator):
                total += self.trainingX[i] 
                num_item += 1.0
        
        return total / num_item
    
    def GetTrainingSigma(self, indicator):       
        num_item = 0.0
        total= 0.0
        
        for i in range( len(self.trainingX)):
            if self.IndicateFuc(i, indicator):
                total += ( self.trainingX[i] - self.GetTrainingMean(indicator) )**2 
                num_item += 1.0
                
        return total / num_item
    
    
    def GetTrainingAlpha(self, classNum):
    # Return a probability of class(classNum) in the training Data set.   
        itemNum = 0
        
        for i in range( len( self.trainingY)):
            if self.IndicateFuc( i, classNum ):
                itemNum += 1
                
        alpha = float( itemNum) / float( len( self.trainingY))
        
        return alpha
    
    def GetMemberFucTesting(self, itemNum, indicator, mean, sigma, alpha ):
        # Return a member function value by testing data
        a = float( math.log(1/ sigma))
        b = float( ( self.testingX[itemNum] - mean)**2 /
                   ( 2*( sigma ) ) ) 
        c = float( math.log( alpha))
        
        return -a-b+c
    
    

       
if __name__ == "__main__":
    _1D2C()
