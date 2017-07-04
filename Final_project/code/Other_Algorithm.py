import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import os

def Read_data():
    
#     option = raw_input ("Choose the data:( 1= sep_svm.data, 2= no_linear_sep.data, 3= circle.data::")
#     if int(option) == 1:
#         f = open('sep_svm.data', 'r')
    filename = raw_input( "Enter the file name:")
    f = open(filename,'r')
    
        
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
        
        extract_x.shape = size, -1
        extract_y.shape = -1, 1
        
    print "==================================================="     
    print "Data Read finishing..."
    print "data with m examples n features:", extract_x.shape, "(m*n)" 
    print "          m examples of label  :", extract_y.shape, "(m*1)"
    print "==================================================="
    
    return extract_x, extract_y

#=====================================================================

def Decision_Tree_Test( X, y):
    
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
    
    n_example_test, n_feature_test =X_test.shape
    
    clf = tree.DecisionTreeRegressor()
    clf = clf.fit(X_train, y_train)
    
    n_error = 0.0
    
    for i in range(n_example_test):
        if int(clf.predict(X_test[i]) ) != int(y_test[i]):
            n_error += 1
    
    
    acc_rate = 1-(n_error/ n_example_test)
    '''
    # clf.predict: return label(class) of input X (with 1*n_feature) info. ;
    print clf.predict([[0., 1., 0., 1.]])
    
    print "Test of only X4 feature..."
    print clf.predict(X[:5,:])
    print clf.predict(X[:5,:]).shape
    
    
    # clf.predict_proba : return a array( 1* n_label) with probability of each class 
    #                     of input X (with 1*n_feature) info. ;
    print clf.predict_proba([[6.3, 3.7, 3.35, 2.0]])
    '''
    return acc_rate

def Decision_Tree_nFold( X, y):
    from sklearn.cross_validation import KFold
    kf = KFold( len(X) , n_folds= 10,shuffle=True, random_state=42)
    sum_acc = 0.0
    Max_acc = 0.0001
    min_acc = 0.9999
    
    
    for train_index, test_index in kf:
        # allocate training and test data
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        
        clf = tree.DecisionTreeRegressor()
        clf = clf.fit(X_train, y_train)
        
        n_example_test, n_feature_test =X_test.shape    
        n_error = 0.0
    
        for i in range(n_example_test):
            if int(clf.predict(X_test[i]) ) != int(y_test[i]):
                n_error += 1
                
                
        acc_rate = 1-(n_error/ n_example_test)
        sum_acc += acc_rate
        
        if acc_rate > Max_acc:
            Max_acc = acc_rate
        if acc_rate < min_acc:
            min_acc = acc_rate
        
        
    return sum_acc/ 10, Max_acc, min_acc
    
#====================================================
def Random_Forest( X, y):
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
    
    n_example_test, n_feature_test =X_test.shape
    
    clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(X_train, y_train)
    
    n_error = 0.0
    
    for i in range(n_example_test):
        if int(clf.predict(X_test[i]) ) != int(y_test[i]):
            n_error += 1
    
    
    acc_rate = 1-(n_error/ n_example_test)
    
    return acc_rate

def Random_Forest_nFold( X, y):
    from sklearn.cross_validation import KFold
    kf = KFold( len(X) , n_folds= 10,shuffle=True, random_state=42)
    sum_acc = 0.0
    Max_acc = 0.0001
    min_acc = 0.9999
    
    
    for train_index, test_index in kf:
        # allocate training and test data
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        
        clf = RandomForestClassifier(n_estimators=10)
        clf = clf.fit(X_train, y_train)
        
        n_example_test, n_feature_test =X_test.shape    
        n_error = 0.0
    
        for i in range(n_example_test):
            if int(clf.predict(X_test[i]) ) != int(y_test[i]):
                n_error += 1
                
                
        acc_rate = 1-(n_error/ n_example_test)
        sum_acc += acc_rate
        
        if acc_rate > Max_acc:
            Max_acc = acc_rate
        if acc_rate < min_acc:
            min_acc = acc_rate
        
        
    return sum_acc/ 10, Max_acc, min_acc
    

if __name__ == "__main__": 
    X, y = Read_data()
    
    acc_rate = Decision_Tree_Test( X, y)
    
    print "Accuracy :", acc_rate, "(0.33 data as test data)"
    
    fold_acc_rate, fold_Max_acc, fold_min_acc = Decision_Tree_nFold( X, y)
    
    div = (fold_Max_acc-fold_acc_rate + fold_acc_rate - fold_min_acc)/2
    print "10Fold validation Accuracy:", fold_acc_rate, "( Max:", fold_Max_acc,",min:", fold_min_acc,")"        
    print "====>", fold_acc_rate*100,"+-", div*100
    print "===================================================="
    
    acc_rate_RF = Random_Forest(X, y)
    print "Accuracy :", acc_rate_RF, "(0.33 data as test data)"
    
    fold_acc_rate_RF, fold_Max_acc_RF, fold_min_acc_RF = Random_Forest_nFold(X, y)
    
    div = (fold_Max_acc_RF - fold_min_acc_RF)/2
    print "10Fold validation Accuracy:", fold_acc_rate_RF, "( Max:", fold_Max_acc_RF,",min:", fold_min_acc_RF,")"        
    print "====>", fold_acc_rate_RF*100,"+-", div*100
    
    
    
    
    
    
    
    
    
    
    
    