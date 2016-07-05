# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


import time
import csv

import numpy as np

start_time = time.clock()

################ Reading ####################################

y_train=np.loadtxt('TrainLabels.csv',delimiter=',', skiprows=1,usecols=[1])
X_train=np.loadtxt('X_train_filtered_channel.csv', delimiter=",")
X_test=np.loadtxt('X_test_filtered_channel.csv',delimiter=",")
y_test = None

n_test = 1000
X_test = X_train[:n_test]
y_test = y_train[:n_test]
X_train = X_train[n_test:]
y_train = y_train[n_test:]

################################### Training
# The main type of model is the Sequential model, a linear stack of layers
# Generally, you need a network large enough to capture the structure of the problem if that helps at all
# in this example we will use a fully-connected network structure with three layers.
# Fully connected layers are defined using the Dense class
# initialize the network weights to a small random number generated from a uniform distribution
# sigmoid and tanh activation functions were preferred for all layers
# better performance is seen using the rectifier activation function
# better performance is seen using the rectifier activation function
model = Sequential()
model.add(Dense(12, input_dim=X_train.shape[1], init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))


optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# Compile model
# specify the loss function to use to evaluate a set of weights
# logarithmic loss, which for a binary classification problem
# efficient gradient descent algorithm adam for no other reason that it is an efficient default
# because it is a classification problem, we will collect and report the classification accuracy as the metric
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# plot(model, to_file='model.png')
# Fit the model
## fixed number of iterations through the dataset called epochs
## set the number of instances that are evaluated before a weight update in the network is performed called the batch size
model.fit(X_train, y_train, nb_epoch=1000, batch_size=100)



if y_test != None:
     scores = model.evaluate(X_test, y_test)
     print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
else:
     pred = model.predict_proba(X_test)
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
