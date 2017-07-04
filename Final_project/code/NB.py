import math
import os
import copy
import numpy as np
import random

lam = 2 #Lambda
learningRate = 0.02

def indicator(Y):
    if Y == 0.5:
        return 1
    else:
        return 0
def getAlpha(trainSize, numTotalClass, numFeature, sizeFeature, VoF, trainX, trainY):
    alpha = np.zeros(numTotalClass)
    alphaGivenY = []
    for i in range(numTotalClass):
        tmp = []
        for i in range(numFeature):
            tmp.append(np.zeros(sizeFeature[i]))
        alphaGivenY.append(tmp)
    for i in range(trainSize):
        tmpC = trainY[i]
        alpha[tmpC] = alpha[tmpC] + 1
        for j in range(numFeature):
            tmpV = VoF[j].index(trainX[i][j])
            alphaGivenY[tmpC][j][tmpV] = alphaGivenY[tmpC][j][tmpV] + 1
    for i in range(numTotalClass):
        for j in range(numFeature):
            alphaGivenY[i][j] = alphaGivenY[i][j] + 0.00001
    for k in range(numTotalClass):
        for i in range(numFeature):
            alphaGivenY[k][i] = alphaGivenY[k][i] * 1.0 / alpha[k]
        alpha[k] = alpha[k] * 1.0 / trainSize
#     print alpha
#     print alphaGivenY
    return alpha, alphaGivenY
def classify(numFeature, numTotalClass, alpha, alphaGivenY, x):
    C = np.zeros(numTotalClass)
    for c in range(numTotalClass):
        tmp = 1.0
        for i in range(numFeature):
            tmpV = VoF[i].index(x[i])
            tmp = tmp * alphaGivenY[c][i][tmpV]
#             print 'x[i] = ', x[i], " alphaGivenY[c][i][tmpV] = ", alphaGivenY[c][i][tmpV], " w[i][tmpV] =", w[i][tmpV]
        C[c] = alpha[c] * tmp
    return np.argmax(C)
def getNumError(testX, testY, numFeature, numTotalClass, alpha, alphaGivenY):
    num = 0
    for i in range(0, len(testY)):
        predicatedY = classify(numFeature, numTotalClass, alpha, alphaGivenY, testX[i])
#         print predicatedY
        if testY[i] != predicatedY:
            num += 1
    return num
def _10_cross(inX, inY, inSize, numTotalClass, numFeature, sizeFeature, VoF):
    errTot = 0
    size = len(inX)/10
    trainSize = inSize - size
    maxError = 0
    minError = size
    for i in range(0,10):
        print "-------------------------------------------------------"
        print i+1, " times:"
        testX = inX[i*size:(i+1)*size]
        testY = inY[i*size:(i+1)*size]
        trainX = inX[:i*size] + inX[(i+1)*size:]
        trainY = inY[:i*size] + inY[(i+1)*size:]
        
        alpha, alphaGivenY = getAlpha(trainSize, numTotalClass, numFeature, sizeFeature, VoF, trainX, trainY)

        
        tmp = getNumError(testX, testY, numFeature, numTotalClass, alpha, alphaGivenY)
        maxError = max(tmp, maxError)
        minError = min(tmp, minError)
        errTot += tmp
        
    print "----10-fold cross-validation----"
    errTot /= 10
    maxAcc = 1 - minError * 1.0 / size
    minAcc = 1 - maxError * 1.0 / size
    print "Maximal accuracy rate = ", maxAcc
    print "Minimal accuracy rate = ", minAcc
    print "Average accuracy rate = ", 1 - errTot * 1.0 / size, " +- ", (maxAcc - minAcc) / 2


def dataShuffle(inX, inY):
    combined = zip(inX, inY)
    random.shuffle(combined)
    return zip(*combined)    

if __name__ == "__main__":
    filename = raw_input( "Enter the file name:")
    fin = open(filename,'r')
    inX = []
    inY = []
    sizeFeature = []
    
    for line in fin.readlines():
        if line[0] != '#':
            temp = map(float,line.split(","))
            inY.extend([int(temp[0])])
            inX.append(temp[1:])
    fin.close()
    
    inX[:], inY[:] = dataShuffle(inX, inY)
    numTotalClass = len(set(inY))
    inSize = len(inY)
    numFeature = len(inX[0])
    
    
    
    VoF = []
    for i in range(numFeature):
        tmp = []
        for j in range(inSize):
            tmp.append(inX[j][i])
        VoF.append(list(set(tmp)))
#     VoF = np.array(VoF)
    for i in VoF:
        sizeFeature.append(len(i))
    
    
    _10_cross(inX, inY, inSize, numTotalClass, numFeature, sizeFeature, VoF)
    
    
        
    