# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense

import time
import csv

import numpy as np

start_time = time.clock()
################ read in training labels ####################################
labels = np.zeros(5440)
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
##################################### Read in train feature matrix X_train ##################################
'''
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
##################################### Read in test feature matrix X_train ##################################
'''
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
################################### Training
'''

# X_test = X_train[0:30, :]
# y_test = y[0:30]

# X_train = X_train[31:-1,:]
# y = y[31:-1]
# Train = numpy.loadtxt("dataset.csv", delimiter=",")
# X = dataset[:,0:8]
# Y = dataset[:,8]
# X = [[1,2,3,4,5,6,7,8],[2,2,3,4,5,6,7,8],[3,2,3,4,5,6,7,8]]
# Y = [[1],[2],[3]]

# The main type of model is the Sequential model, a linear stack of layers
# Generally, you need a network large enough to capture the structure of the problem if that helps at all
# in this example we will use a fully-connected network structure with three layers.
# Fully connected layers are defined using the Dense class
# initialize the network weights to a small random number generated from a uniform distribution
# sigmoid and tanh activation functions were preferred for all layers
# better performance is seen using the rectifier activation function
# better performance is seen using the rectifier activation function
model = Sequential()
model.add(Dense(12, input_dim=260, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
# specify the loss function to use to evaluate a set of weights
# logarithmic loss, which for a binary classification problem
# efficient gradient descent algorithm adam for no other reason that it is an efficient default
# because it is a classification problem, we will collect and report the classification accuracy as the metric
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
## fixed number of iterations through the dataset called epochs
## set the number of instances that are evaluated before a weight update in the network is performed called the batch size
model.fit(X_train, y, nb_epoch=1, batch_size=2)

# evaluate the model
# scores = model.evaluate(X_test, y_test)
pred = model.predict_classes(X_test)
# pred = model.prediction(X_test)

# pred = pred[:,1]

# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

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
