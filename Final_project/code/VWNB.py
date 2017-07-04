import math
import os
import copy
import numpy as np

lam = 2 #Lambd


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
def training(N, trainSize, numTotalClass, numFeature, sizeFeature, VoF, trainX, trainY, alpha, alphaGivenY, learningRate):
    global lam
    
    w = []
    O = np.zeros((trainSize, numTotalClass))
    T = []
    
    for i in sizeFeature:
        w.append(np.ones(i) * 1)
       
    for i in range(trainSize):
        tmp = np.zeros(numTotalClass)
        tmp[trainY[i]] = 0.5
        T.append(tmp)
    
    maxAlpha = np.amax(alpha)
    maxAlphaGivenY = alphaGivenY[np.argmax(alpha)]
    
    t = 1
    change = True
    while change :
        change = False
        print '.',
        if t %50 == 0 :
            print " " 
#         print "w = ", w
#         for each training data Dk do 
        
        for i in range(numFeature):
            for j in range(sizeFeature[i]):
                for l in range(numTotalClass):
                    
                    for k in range(trainSize):
                        tmp1 = 0.0
                        for a in range(numFeature):
                            tmpV = VoF[a].index(trainX[k][a])
                            tmp1 = tmp1 + w[a][tmpV] * math.log(alphaGivenY[l][a][tmpV] / maxAlphaGivenY[a][tmpV])
#                         tmp = 1.0 / (1 + math.exp(math.log(maxAlpha / alpha[l]) + tmp1))
    #                         if tmp > 0.5:
    #                             print 1.0 / (1 + math.exp(math.log(maxAlpha / alpha[l]) + tmp1))
                        O[k][l] = 1.0 / (1 + math.exp(math.log(maxAlpha / alpha[l]) + tmp1))
                tmp2 = 0.0
                for a in range(trainSize):
                    for b in range(numTotalClass):
                        tmp2 = tmp2 + (O[a][b] - T[a][b]) * O[a][b] * (1 - O[a][b]) * math.log(alphaGivenY[b][i][j] / maxAlphaGivenY[i][j])
                wNew = w[i][j] * (1.0 - learningRate**lam) - learningRate * tmp2
                if abs(wNew - w[i][j]) > 0:
                    change = True
                w[i][j] = wNew
        learningRate = learningRate * 1.0 / (1 + t * 1.0 / N)
        t = t + 1
    print " "
    return w
def classify(numFeature, numTotalClass, alpha, alphaGivenY, w, x):
    C = np.zeros(numTotalClass)
    for c in range(numTotalClass):
        tmp = 1.0
        for i in range(numFeature):
            tmpV = VoF[i].index(x[i])
            tmp = tmp * alphaGivenY[c][i][tmpV]**w[i][tmpV]
#             print 'x[i] = ', x[i], " alphaGivenY[c][i][tmpV] = ", alphaGivenY[c][i][tmpV], " w[i][tmpV] =", w[i][tmpV]
        C[c] = alpha[c] * tmp
    return np.argmax(C)
def getNumError(testX, testY, numFeature, numTotalClass, alpha, alphaGivenY, w):
    num = 0
    for i in range(0, len(testY)):
        predicatedY = classify(numFeature, numTotalClass, alpha, alphaGivenY, w, testX[i])
#         print predicatedY
        if testY[i] != predicatedY:
            num += 1
    return num
def _10_cross(inX, inY, inSize, numTotalClass, numFeature, sizeFeature, VoF, learningRate):
    errTot = 0
    size = len(inX)/10
    trainSize = inSize - size
    maxError = 0
    minError = size
    learningRateSave =  learningRate
    for i in range(0,10):
        learningRate = learningRateSave
        print "-------------------------------------------------------"
        print i+1, " times:"
        testX = inX[i*size:(i+1)*size]
        testY = inY[i*size:(i+1)*size]
        trainX = inX[:i*size] + inX[(i+1)*size:]
        trainY = inY[:i*size] + inY[(i+1)*size:]
        
        alpha, alphaGivenY = getAlpha(trainSize, numTotalClass, numFeature, sizeFeature, VoF, trainX, trainY)
        w = training(inSize, trainSize, numTotalClass, numFeature, sizeFeature, VoF, trainX, trainY, alpha, alphaGivenY, learningRate)
        
        tmp = getNumError(testX, testY, numFeature, numTotalClass, alpha, alphaGivenY, w)
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
    

if __name__ == "__main__":
    filename = raw_input( "Enter the file name:")
    fin = open(filename,'r')
    inX = []
    inY = []
    sizeFeature = []
    
    learningRate = float( raw_input( "Set learning rate( if learning rate > 0.1 will set to 0.1):") )
    if float(learningRate) > 0.1:
        learningRate = 0.1
        print "Set learning rate = 0.1"
    
    for line in fin.readlines():
        if line[0] != '#':
            temp = map(float,line.split(","))
            inY.extend([int(temp[0])])
            inX.append(temp[1:])
#             inY.extend(map(int, temp[-1:]))
    fin.close()
    

    numTotalClass = len(set(inY))
    inSize = len(inY)
    
    numFeature = len(inX[0])
    
    
    
    VoF = []
    for i in range(numFeature):
        tmp = []
        for j in range(inSize):
            tmp.append(inX[j][i])
        VoF.append(list(set(tmp)))
    for i in VoF:
        sizeFeature.append(len(i))
    
    
    

    
    
    _10_cross(inX, inY, inSize, numTotalClass, numFeature, sizeFeature, VoF, learningRate)
    
    
        
    