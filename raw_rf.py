__author__ = 'Nicolas'


import gzip
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
import cPickle
from random import *
import glob
import csv
import time

start_time = time.clock()

################ read in training labels ####################################
'''labels = np.zeros(5440)
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
y=np.loadtxt('TrainLabels.csv',delimiter=',', skiprows=1,usecols=[1])

'''
##################################### Read in train feature matrix X_train ##################################

X_train = np.zeros((5440,260))
with open('X_train.csv', 'rb') as csvfile:
         trial_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
         rownum = 0
         for row in trial_reader:
            colnum=0
            for col in row:
                X_train[rownum-1][colnum-1]=row[colnum]
                colnum+=1
            rownum+=1
 '''
X_train=np.loadtxt('X_train.csv', delimiter=",")


'''
##################################### Read in test feature matrix X_train ##################################

X_test = np.zeros((3400,260),dtype=np.float)
with open('X_test.csv', 'rb') as csvfile:
         trial_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
         rownum = 0
         for row in trial_reader:
            colnum=0
            for col in row:
                X_test[rownum][colnum]=row[colnum]
                colnum += 1
            rownum += 1
'''
X_test=np.loadtxt('X_test.csv',delimiter=",")

'''
#### Now train a simple SVM classifier ####
'''
print('X_train: ', X_train[0:1,0:1],np.shape(X_train))
print('X_test: ', X_test[0:1,0:1],np.shape(X_test))

clf = RandomForestClassifier(n_estimators=700)
clf.fit(X_train,y)
#clf.fit(X_train[0:5000,:],y[0:5000])


pred = clf.predict_proba(X_test)
pred = pred[:,1]


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

end_time = time.clock()
print'time elapse: ',(end_time-start_time)/60,' minutes'