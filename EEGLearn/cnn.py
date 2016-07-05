# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils


import time
import csv

import numpy as np

import utils

# from keras.utils.visualize_util import plot

start_time = time.clock()

################ Reading ####################################
X_train, y_train, X_test = utils.load_data("Data.mat")
y_train = y_train + 1                     # Don't ask

y_test = None

size = 128,128
from PIL import Image
mg = Image.fromarray(X_train[0], 'RGB')
img.show()
img = img.resize(size, Image.ANTIALIAS)
img.save("resized_image.png")
# n_test = 1000
# X_test = X_train[:n_test]
# y_test = y_train[:n_test]
# X_train = X_train[n_test:]
# y_train = y_train[n_test:]

print(X_train[0].shape)

# Create the model
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(3, 16, 16), activation='relu', border_mode='same'))
model.add(Dropout(0.2))
# model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
# model.add(Dropout(0.2))
# model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
# model.add(Dropout(0.2))
# model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
# model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
# model.add(Dropout(0.2))
# model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='softmax'))
# Compile model
epochs = 1000
lrate = 0.1
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

if y_test != None:
    model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=epochs, batch_size=64)
else:
    model.fit(X_train, y_train, nb_epoch=epochs, batch_size=64)
# Final evaluation of the model


if y_test != None:
     scores = model.evaluate(X_test, y_test)
     print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
else:
     pred = model.predict_proba(X_test)
     pred = pred[:,0]
     
     print(pred[0:10])
     
     
     output = np.zeros(3401,dtype=object)
     with open('../SampleSubmission.csv', 'rb') as csvfile:
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
