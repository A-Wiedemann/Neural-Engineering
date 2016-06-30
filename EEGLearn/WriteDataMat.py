import EEGrunt

source = 'openbci'

path = '../train/'

# EEG data file name
filename = 'Data_S02_Sess01.csv'

# Session title (used in plots and such)
session_title = "Data_S02_Sess01"

# Initialize
EEG = EEGrunt.EEGrunt(path, filename, source, session_title)


# Here we can set some additional properties
# The 'plot' property determines whether plots are displayed or saved.
# Possible values are 'show' and 'save'
EEG.plot = 'save'

EEG.load_data()

# # print(EEG.channels)
# EEG.load_channel(1)
# print(EEG.bandpass(8,12))
# EEG.remove_dc_offset()
# EEG.notch_mains_interference()
# EEG.signalplot()
# EEG.get_spectrum_data()
# EEG.spectrogram()
# EEG.plot_band_power(8,12,"Alpha")
# EEG.plot_spectrum_avg_fft()



# list_of_train_files = glob.glob('../train/Data_*.csv')
# print('list_of_train_files: ', list_of_train_files[0:5])
# list_of_train_files = ['../train/Data_S02_Sess01.csv']

# X_train=[]
# file_it=1

# for files in list_of_train_files:
#     data = np.zeros((300000,56))
#     EOG = np.zeros(300000)
#     feedback = np.zeros(300000)
#     with open(files, 'rb') as csvfile:
#          trial_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
#          rownum = 0
#          for row in trial_reader:
#              #print(row[0])
#              if rownum>0:
#                 colnum=0
#                 for col in row:
#                     if colnum>0 and colnum<57 : # cut of the time column
#                         data[rownum-1][colnum-1]=row[colnum]
#                     if colnum == 57:
#                         EOG[rownum-1]=row[colnum]
#                     if colnum == 58:
#                         feedback[rownum-1]=row[colnum]
#                     colnum += 1
#              rownum += 1


#     data_shape =  np.shape(data)
#     print('Size of data: ',data_shape)

#     #print(data[0:1,0:1])
#     print('EOG: ', EOG[0:5])
#     print('feedback: ', feedback[0:5])
#     #print('Size of reshaped data_vector: ', np.shape(data_vector))

#     events = np.where(feedback == 1)
#     events = np.reshape(events,(-1,1))
#     print('events: ' ,events[0:2], np.shape(events))


#     X_new_file = []
#     it = 1
#     for feedback_event in events:
#         feature_matrix = data[feedback_event : feedback_event+260, :]
        
#         # feature_matrix = np.transpose(feature_matrix)
#         # feature_vector = feature_matrix.ravel()
#         # feature_vector = data[feedback_event : feedback_event+260, 29]  # channel CP2 is number 40, Cz is 30
#         # np.transpose(feature_vector)
#         if it==1:
#             X_new_file = feature_vector
#             #print(feature_vector, np.shape(feature_vector))
#         else:
#             X_new_file= np.vstack((X_new_file,feature_vector))
#         it+=1
#     #print('data X: ',X_new_file[0:3,0:5], np.shape(X_new_file), it)

#     if file_it == 1:
#         X_train = X_new_file
#     else:
#         X_train = np.vstack((X_train,X_new_file))
#     file_it+=1
#     print('shape of X_train: ', np.shape(X_train))

# ############### Now write X_test to a csv file ###################
# '''
# with open('X_train.csv','wb') as csvfile:
#     result_writer = csv.writer(csvfile,delimiter=',')

#     X_shape = np.shape(X_train)         # 3400 x 260
#     for i in xrange(X_shape[0]):
#         for j in xrange(X_shape[1]) :
#             #result_writer.writerow([output[i],pred[i-1]])
#             result_writer.write([X_test[i,j]])
# '''
# np.savetxt('X_train.csv', X_train, delimiter=",")
