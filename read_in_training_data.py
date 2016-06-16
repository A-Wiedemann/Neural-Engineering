__author__ = 'Nicolas'

import gzip
import numpy as np
import time
from sklearn import svm, grid_search
import cPickle
from random import *




############### Write the resulting predictions in a csv file #####################
import csv


# read in training labels
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

# read in training data
data = np.zeros((132001,56))
EOG = np.zeros(132001)
feedback = np.zeros(132001)
with open('train/Data_S02_Sess01.csv', 'rb') as csvfile:
     trial_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
     rownum = 0
     for row in trial_reader:
         #print(row[0])
         if rownum>0:
            colnum=0
            for col in row:
                if colnum>0 and colnum<57 : # cut of the time column
                    data[rownum-1][colnum-1]=row[colnum]
                if colnum == 57:
                    EOG[rownum-1]=row[colnum]
                if colnum == 58:
                    feedback[rownum-1]=row[colnum]
                colnum += 1
         rownum += 1

data_shape =  np.shape(data)
print('Size of data: ',data_shape)

print(data[0:1,0:1])
# reshape appends the rows
#data_vector = np.reshape(data,7656058)
#print('data: ', data_vector[0:5])
#print('data: ', data[0:2][0:2])
print('EOG: ', EOG[0:5])
print('feedback: ', feedback[0:5])
#print('Size of reshaped data_vector: ', np.shape(data_vector))

events = np.where(feedback == 1)
events = np.reshape(events,(-1,1))
print('events: ' ,events[0:5], np.shape(events))


X = []
it = 1
for feedback_event in events:

    #print(feedback_event)
    #event_index = events[feedback_event]
    # if index+260 < data_shape[0]
    #feature_vector = data[event_index+1:event_index+260][39]                           # channel CP2 is number 40
    feature_vector = data[feedback_event : feedback_event+260, 39]
    np.transpose(feature_vector)
    if it==1:
        X = feature_vector
        print(feature_vector, np.shape(feature_vector))
    else:
        X= np.vstack((X,feature_vector))
    it+=1
print('data X: ',X[0:3][0:5], np.shape(X), it)
y = labels[0:60]

'''
###################################### Read in test data and produce the test matrix #########################
'''


'''
#### Now train a simple SVM classifier ####
'''
clf = svm.SVC(kernel="linear", C=0.01, gamma=0.001)

# GRID SEARCH for finding the best parameters
#svr = svm.SVC()
#clf = grid_search.GridSearchCV(svr,param_grid)
clf.fit(X,y)


pred = clf.predict(X)

print(y[0:10])
print(pred[0:10])
