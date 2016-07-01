import EEGrunt
import glob
import csv
import numpy as np
import scipy.io as sio

source = 'bci-challenge'

list_of_train_files = glob.glob('../train/Data_*.csv')

X_train=[]
file_it=1

for csvfile in list_of_train_files:

    print('Loading csvfile: ',csvfile)
    EOG = np.loadtxt(csvfile, delimiter=',',skiprows=1,usecols=[57])
    feedback = np.loadtxt(csvfile, delimiter=',',skiprows=1,usecols=[58])
    data = np.loadtxt(csvfile, delimiter=',',skiprows=1,usecols=range(0,57))

    events = np.where(feedback == 1)
    events = np.reshape(events,(-1,1))

    X_new_file = []
    it = 1

    for feedback_event in events:
        feature_matrix = data[feedback_event : feedback_event+260, :]
        EEG = EEGrunt.EEGrunt("", "", source, "")
        EEG.set_data(feature_matrix)
        # EEG.plot = 'show'
        feature_vector = []

        for channel in EEG.channels:
            EEG.load_channel(channel)
            theta = np.mean(EEG.bandpass(4,8))
            feature_vector.append(theta)

        for channel in EEG.channels:
            EEG.load_channel(channel)
            alpha = np.mean(EEG.bandpass(8,12))
            feature_vector.append(alpha)

        for channel in EEG.channels:
            EEG.load_channel(channel)
            beta = np.mean(EEG.bandpass(13,30))
            feature_vector.append(beta)

        if it==1:
            X_new_file = feature_vector
            #print(feature_vector, np.shape(feature_vector))
        else:
            X_new_file= np.vstack((X_new_file,feature_vector))
        it+=1
    #print('data X: ',X_new_file[0:3,0:5], np.shape(X_new_file), it)

    if file_it == 1:
        X_train = X_new_file
    else:
        X_train = np.vstack((X_train,X_new_file))
    file_it+=1
    print('shape of X_train: ', np.shape(X_train))

############### Now write X_test to a mat file ###################
labels = np.loadtxt('../TrainLabels.csv', delimiter=',',skiprows=1,usecols=[1])

# np.savetxt('X_train.csv', X_train, delimiter=",")
sio.savemat('Data.mat',{'featMat': X_train, 'labels': labels})
