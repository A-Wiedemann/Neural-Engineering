__author__ = 'Nicolas'

import gzip
import numpy as np
import time
from sklearn import svm, grid_search
import cPickle
from random import *


pred = np.random.randint(2, size=3400)                                              # generate a random vector of 1's and 0's

############### Write the resulting predictions in a csv file #####################
import csv

# read in the submission labels and store them in the output variable
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
