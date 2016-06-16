__author__ = 'Nicolas'

__author__ = 'Nicolas'

import gzip
import numpy as np
import time
from sklearn import svm, grid_search
import cPickle
from random import *
import glob
import csv


################ read in training labels ####################################
labels = np.zeros(5440,dtype=object)
with open('TrainLabels.csv', 'rb') as csvfile:
     trial_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
     rownum = 0
     for row in trial_reader:
         #print(row[0])
         if rownum>0:
            labels[rownum-1]=row[1]
         rownum += 1

print('Labels: ',labels[0:10])
y = labels

'''
##################################### Read in train feature matrix X_train ##################################
'''
X_train = np.zeros((5440,260))
with open('X_train.csv', 'rb') as csvfile:
         trial_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
         rownum = 0
         for row in trial_reader:
             #print(row[0])
             if rownum>0:
                colnum=0
                for col in row:
                    X_train[rownum-1][colnum-1]=row[colnum]
'''
##################################### Read in test feature matrix X_train ##################################
'''
X_test = np.zeros((3400,260))
with open('X_train.csv', 'rb') as csvfile:
         trial_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
         rownum = 0
         for row in trial_reader:
             #print(row[0])
             if rownum>0:
                colnum=0
                for col in row:
                    X_test[rownum-1][colnum-1]=row[colnum]

'''
#### Now train a simple SVM classifier ####
'''
print('X_train: ', np.shape(X_train))
print('X_test: ', np.shape(X_test))

clf = svm.SVC(kernel="linear", C=50, gamma=0.01)
#clf = svm.SVC()
# GRID SEARCH for finding the best parameters
#svr = svm.SVC()
#clf = grid_search.GridSearchCV(svr,param_grid)
clf.fit(X_train,y)


pred = clf.predict(X_test)

print(pred[0:10])


output = np.zeros(3401,dtype=object)
with open('SampleSubmission.csv', 'rb') as csvfile:
     trial_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
     rownum = 0
     for row in trial_reader:
         #print(row[0])
         if rownum>0:
            output[rownum]=row[0]
         rownum += 1

print(output[0:10])


# now open the new submission file and write in the labels and the data
with open('predictions.csv','wb') as csvfile:
    result_writer = csv.writer(csvfile,delimiter=',')
    result_writer.writerow(['IdFeedBack','Prediction'])

    for i in range(1,3401):
        result_writer.writerow([output[i],pred[i-1]])