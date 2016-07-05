__author__ = 'Nicolas'


import gzip
import numpy as np
import time
from sklearn.ensemble import GradientBoostingClassifier
import cPickle
from random import *
import glob
import csv
import time

start_time = time.clock()

################ read in training labels ####################################
y=np.loadtxt('TrainLabels.csv',delimiter=',', skiprows=1,usecols=[1])
X_train=np.loadtxt('X_train_filtered_channel.csv', delimiter=",")
X_test=np.loadtxt('X_test_filtered_channel.csv',delimiter=",")




'''
#### Now train a simple SVM classifier ####
'''



#print('X_train: ', X_train[0:1,0:1],np.shape(X_train))
#print('X_test: ', X_test[0:1,0:1],np.shape(X_test))


clf = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05).fit(X_train, y)

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