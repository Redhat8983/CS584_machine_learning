'''
@author: Yung Chi Liu
'''
import matplotlib.pyplot as plt
import numpy as np
from pylab import *

class NB_Binominal():
    def __init__(self):
        self.allDataX = np.array([])
        self.allDataY = np.array([]) 
        
        self.trainingX = np.array([])
        self.testingX = np.array([])
        self.trainingY = np.array([])
        self.testingY = np.array([])
        
        self.bigPtraining = []
        self.bigPtesting =[]
        
        self.SizeOfAllData = 0
        self.featureNum = 0
        
        self.InitReadData()
        
        
    # Main program()
    def RunBody(self):
        
        foldNum = 10
        self.featureNum = len( self.allDataX)/ self.SizeOfAllData  
        
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
            
            self.bigPtraining = sum( self.trainingX, axis = 1)
            self.bigPtesting  = sum( self.testingX, axis = 1)
        
            alpha_zero = self.GetTrainingAlpha(0)
            alpha_1st = self.GetTrainingAlpha(1)
            
            class_0_Num = 0
            class_1_Num = 0 
            for i in range( len( self.trainingY)):
                if int( self.trainingY[i]) == int(0):
                    class_0_Num +=1
                else:
                    class_1_Num +=1
       
            
            # compute alpha(j)| y=i: j-th feature, i-th class:
            # alpha(j)|y=i look like: [ F-1 of 1 , F-2 of 1
            # get a list[] of (# of 1 in each feature.)/ total_class j num
            
            alpha_base_0 = self.GetTrainingAlpaBaseOn( 0, class_0_Num)
            alpha_base_1 = self.GetTrainingAlpaBaseOn( 1, class_1_Num)
            
            TP = 0.0
            FN = 0.0
            TN = 0.0
            FP = 0.0
            
#             print "Testing", self.testingX
                         
            # start discriminate function 
            for i in range( len( self.testingY)):
                g0 = self.GetMemberFucTesting(i, alpha_zero, alpha_base_0 )
                g1 = self.GetMemberFucTesting(i, alpha_1st, alpha_base_1 )
                #print g0 ," --", g1
                
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
        print "Testing Size::", self.SizeOfAllData/foldNum
        
        print foldNum, "-fold validation"
        print "   P   N"
        print "P ", totalTP, "  ", totalFP
        print "N ", totalFN, "  ", totalTN
        print " "  
        
        print "Precision =", precision/foldNum
        print "Recall =", recall/foldNum
        print "Accuracy =", accuracy/foldNum
        print "F-maesure= ", F_maesure/foldNum

    def GetTrainingAlpaBaseOn(self, indicator, total ):
        
        trainingSum = np.zeros( (1, self.featureNum))
        bigPSum = 0
        
        for item in range( len(self.trainingY) ):
            
            if self.IndicateFuc( item, indicator):
                trainingSum += self.trainingX[item]
                bigPSum += self.bigPtraining[item]

        return (trainingSum ) / (bigPSum )
            
    def GetMemberFucTesting(self, itemNum, alpha, part_alpha ):
        a =0.0
        alineTestingX = np.array( ( self.featureNum))
        alineTestingX = self.testingX[itemNum]
        alineTestingX.tolist()
        
        for i_feature in range( self.featureNum): 
            #  P! / (Xj)! * ( P-Xj)!
            #  a0 /  a1   *  a2    = a3
            a0 = math.factorial( self.bigPtesting[itemNum])
            a1 = math.factorial( alineTestingX[i_feature] )
            a2 = math.factorial( ( self.bigPtesting[itemNum])- ( alineTestingX[i_feature]))
            a3 = a0 / ( a1*a2)          
            # a4
            a4 = ( part_alpha[0][i_feature])**alineTestingX[i_feature]
            a5 = ( 1 - part_alpha[0][i_feature] )**( self.bigPtesting[itemNum]- alineTestingX[i_feature])
            a = math.log( a3* a4 *a5 )
        b = math.log( alpha )
    
        return a+b
    
    def GetTrainingAlpha(self, classNum):
    # Return a probability of class(classNum) in the training Data set.   
        itemNum = 0
        
        for i in range( len( self.trainingY)):
            if self.IndicateFuc( i, classNum ):
                itemNum += 1
                
        alpha = float( itemNum) / float( len( self.trainingY))
        
        return alpha
        
    
    # Load Data to :: allDataX, allDataY
    # Get value    :: SizeOfAllData
    def InitReadData(self):
        switch = 0
        
        while ( switch == 0):
            option = raw_input ("Enter the data:( 1= 5_monk.data ,0 = exit program) ::")
            
            self.allDataX = ([])
            self.allDataY = ([]) 
        
            self.trainingX = ([])
            self.testingX = ([])
            self.trainingY = ([])
            self.testingY = ([])
            
            if ( int(option) == 1):  #start to read data in np.array
                f = open('5_monk.data', 'r')
                
                for line in f.readlines():
                    #split the string on whitespace, return a list of numbers 
                    # (as strings)
                    numbers_str = line.replace('?', '-1').split("," )
                    #convert numbers to floats
                    numbers_float = [float(x) for x in numbers_str]  #map(float,numbers_str) works too
                    
                    self.allDataX = np.append( self.allDataX, numbers_float[1:])
                    self.allDataY = np.append( self.allDataY, numbers_float[0])
                    self.SizeOfAllData = len( self.allDataY)
                    
                print self.SizeOfAllData
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
        
        dataRange = self.SizeOfAllData / total_foldNum
        fetureNum = len( self.allDataX) / self.SizeOfAllData
        
        print "Round ", foldNum, " ", total_foldNum, " Fold, processing..."
        # Init. np.array...
        self.trainingX = ([])
        self.trainingY = ([])  
        self.testingX = ([])
        self.testingY = ([])   
        
        self.bigPtraining = []
        self.bigPtesting =[]
        
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
    NB_Binominal()

