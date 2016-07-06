__author__ = 'Nicolas'

import gzip
import numpy as np
import time
from sklearn import svm
import cPickle
from random import *
import glob
import csv
import time

start_time = time.clock()

################ read in training labels ####################################
y = np.loadtxt('TrainLabels.csv', delimiter=',', skiprows=1, usecols=[1])
X_train = np.loadtxt('X_train_filtered_channel.csv', delimiter=",")
X_test = np.loadtxt('X_test_filtered_channel.csv', delimiter=",")

'''
#### Now train a simple SVM classifier ####
'''



# print('X_train: ', X_train[0:1,0:1],np.shape(X_train))
# print('X_test: ', X_test[0:1,0:1],np.shape(X_test))
clf = svm.SVC(kernel="linear", C=1, gamma=1e-4, probability=True)
# clf = svm.SVC()
# GRID SEARCH for finding the best parameters
# param_grid = {'kernel': ('linear', 'rbf'), 'C': [0.01, 10], 'gamma': [0.00001, 0.01]}
# svr = svm.SVC()
# clf = grid_search.GridSearchCV(svr,param_grid)
X_train = X_train[:, ::3]  # downsampling with a factor of 4
print('downsampled X_train: ', np.shape(X_train))
#clf.fit(X_train, y)
clf.fit(X_train[0:600, :], y[0:600])

X_test = X_test[:, ::3]
pred = clf.predict_proba(X_test)
pred = pred[:, 1]

print(pred[0:10])

output = np.zeros(3401, dtype=object)
with open('SampleSubmission.csv', 'rb') as csvfile:
    trial_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    rownum = 0
    for row in trial_reader:
        # print(row[0])
        if rownum > 0:
            output[rownum] = row[0]
        rownum += 1

print(output[0:10])


# now open the new submission file and write in the labels and the data
with open('predictions.csv', 'wb') as csvfile:
    result_writer = csv.writer(csvfile, delimiter=',')
    result_writer.writerow(['IdFeedBack', 'Prediction'])

    for i in range(1, 3401):
        result_writer.writerow([output[i], pred[i - 1]])

end_time = time.clock()
print'time elapse: ', (end_time - start_time) / 60, ' minutes'
