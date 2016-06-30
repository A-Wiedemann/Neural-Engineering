__author__ = 'Nicolas'

__author__ = 'Nicolas'

import glob
import csv

from pylab import *
from scipy.signal import *
import pandas as pd
import numpy as np


list_of_train_files = glob.glob('test/Data_*.csv')
print('list_of_train_files: ', list_of_train_files[0:5])


def bandpass(signal,band,fs):
    B,A = butter(5, array(band)/(fs/2), btype='bandpass')
    return lfilter(B, A, signal, axis=0)

freq = 200
response_window = 1.3 * freq

data = []
X_train =[]
X_train_average = []
it_file=1

for f in list_of_train_files:
    print f
    signal = np.array(pd.io.parsers.read_csv(f))
    data = signal[:,1:-2]
    #print('data: ', data[0:3,0:3]) works
    EOG = signal[:,-2]
    feedback = signal[:,-1]

    filtered_signal = bandpass(data,[1.0,40.0],freq)

    print('filtered_signal shape: ', np.shape(filtered_signal))
    print('filtered signal', filtered_signal[0:3,0:3])

    events = np.where(feedback == 1)
    events = np.reshape(events,(-1,1))
    #print('events: ' ,events[0:2], np.shape(events))


    it_event = 0
    X_buffer=np.zeros((100,56))
    #X_buffer=[]
    for feedback_event in events:
        for channel in range(0, 56):
            feature_vector = filtered_signal[feedback_event : feedback_event+response_window, channel]
            #print('feature_vector: ', np.shape(feature_vector))
            #print('feature_vector: ', feature_vector[0:5])
            X_buffer[it_event,channel] = np.average(feature_vector,0)
        it_event +=1

    rownum=100
    if it_file % 5 is not 0:
        X_buffer = np.resize(X_buffer, (60,56))
        rownum=60
    print('X_buffer: ', np.shape(X_buffer))
    if it_file==1:
        X_train_average=X_buffer
    else:

        X_train_average = np.vstack((X_train_average,X_buffer))
    #X_train_average.append(X_buffer,axis=0)
    print('X_train_average: ', np.shape(X_train_average))
    it_file+=1

#print('X_train_average: ', np.shape(X_train_average))

np.savetxt('X_test_average.csv', X_train_average, delimiter=",")








