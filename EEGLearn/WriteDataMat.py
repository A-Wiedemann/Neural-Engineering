import EEGrunt
import glob
import numpy as np
import scipy.io as sio
import csv
import sys, getopt
import eeg_cnn_lib
import utils

import pdb




def main(argv):
    outformat = ""
    try:
        opts, args = getopt.getopt(argv,"f:",["format="])
    except getopt.GetoptError:
        print 'WriteDataMat.py -f <format>'
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == '-h':
            print 'WriteDataMat.py -f <format>'
            sys.exit()
        elif opt in ("-f", "--format"):
            outformat = arg
            if outformat not in ["raw", "avg_img"]:
                print '<format> = {raw | avg_img}'
                sys.exit()
                    
    print('Output format is ', outformat)
    
    ############ Calculate position of electrodes ##############################################
    # Radius and angle of the electrodes are given in the file ChannelsLocation.csv
    # The library utils offer a function for conversion the polar coordinates into cartesian
    # coordinates.
    channelsLocation = np.zeros((56,2))
    with open("../Other/ChannelsLocation.csv","r") as csvfile:
        reader = csv.DictReader(csvfile,delimiter = ',')
        rownum = 0
        for row in reader:
            # print(row['Radius'],row['Phi'])
            # print(utils.pol2cart(float(row['Radius']),float(row['Phi'])))
            channelsLocation[rownum] = utils.pol2cart(float(row['Radius']),float(row['Phi']))
            rownum += 1
    print("Channel locations loaded.")
    
    X_train=readX('../train/Data_*.csv', outformat)
    X_test =readX('../test/Data_*.csv', outformat)
    pdb.set_trace()
    features = np.vstack((X_train,X_test))
    
    features = eeg_cnn_lib.gen_images(channelsLocation, features, 16, augment=False, pca=False, n_components=2)
    
    X_train = features[:X_train.shape[0]]
    X_test = features[X_train.shape[0]:]
    
    labels = np.loadtxt('../TrainLabels.csv', delimiter=',',skiprows=1,usecols=[1])  # Such that the labels matches X_train in order the list must be in alphabetical order
    
    
    
    # np.savetxt('X_train.csv', X_train, delimiter=",")
    sio.savemat('Data.mat',{'featMat': X_train, 'labels': labels, 'featMatTest': X_test})


def readX(folder_path, outformat):
    source = 'bci-challenge'
    print("Loading files of: ", folder_path)
    list_of_files = sorted(glob.glob(folder_path))
    X=[]
    
    n_channel = 38              # CP 2
    
    for csvfile in list_of_files:
        
        print('Loading csvfile: ',csvfile)
        EOG = np.loadtxt(csvfile, delimiter=',',skiprows=1,usecols=[57])
        feedback = np.loadtxt(csvfile, delimiter=',',skiprows=1,usecols=[58])
        data = np.loadtxt(csvfile, delimiter=',',skiprows=1,usecols=range(0,57))  # [time ch 1 ... ch 56]
        
        events = sorted(np.where(feedback == 1))
        events = np.reshape(events,(-1,1))
        
        X_new_file = []
        
        for feedback_event in events:
            if outformat == "raw":
                feature_vector = data[feedback_event : feedback_event+260, n_channel]
            elif outformat == "avg_img":
                feature_matrix = data[feedback_event : feedback_event+260, :]
                EEG = EEGrunt.EEGrunt("", "", source, "")
                EEG.set_raw_data(feature_matrix)
                # EEG.plot = 'show'
                feature_vector = []
                
                for channel in EEG.channels:
                    EEG.load_channel(channel)
                    theta = np.mean(EEG.bandpass(4,8))
                    # theta = EEG.bandpass(4,8)
                    feature_vector.append(theta)
                    
                for channel in EEG.channels:
                    EEG.load_channel(channel)
                    alpha = np.mean(EEG.bandpass(8,12))
                    # alpha = EEG.bandpass(8,12)
                    feature_vector.append(alpha)
                    
                for channel in EEG.channels:
                    EEG.load_channel(channel)
                    beta = np.mean(EEG.bandpass(13,30))
                    # beta = EEG.bandpass(13,30)
                    feature_vector.append(beta)
                
            if X_new_file == []:
                X_new_file = feature_vector
            else:
                X_new_file= np.vstack((X_new_file,feature_vector))
                #print('data X: ',X_new_file[0:3,0:5], np.shape(X_new_file), it)
                
        if X == []:
            X = X_new_file
        else:
            X = np.vstack((X,X_new_file))
            print('shape of X_train: ', np.shape(X))
    
    
    
    
    # X = gen_images(channelsLocation, X, 16, augment=False, pca=False, n_components=2)
    
    return X



if __name__ == "__main__":
    main(sys.argv[1:])
