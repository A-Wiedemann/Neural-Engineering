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

print(labels[0:10])

# read in training data
data = np.zeros((132001,58))
with open('test/Data_S01_Sess01.csv', 'rb') as csvfile:
     trial_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
     rownum = 0
     for row in trial_reader:
         #print(row[0])
         if rownum>0:
            colnum=0
            for col in row:
                if colnum>0:
                    data[rownum-1][colnum-1]=row[colnum]
                colnum += 1
         rownum += 1

print(data[0:1])
# reshape appends the rows
data_vector = np.reshape(data,7656058)

