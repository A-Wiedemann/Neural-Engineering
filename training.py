# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense

import numpy

dataset = numpy.loadtxt("dataset.csv", delimiter=",")
# X = dataset[:,0:8]
# Y = dataset[:,8]
X = [[1,2,3,4,5,6,7,8],[2,2,3,4,5,6,7,8],[3,2,3,4,5,6,7,8]]
Y = [[1],[2],[3]]

# The main type of model is the Sequential model, a linear stack of layers
# Generally, you need a network large enough to capture the structure of the problem if that helps at all
# in this example we will use a fully-connected network structure with three layers.
# Fully connected layers are defined using the Dense class
# initialize the network weights to a small random number generated from a uniform distribution
# sigmoid and tanh activation functions were preferred for all layers
# better performance is seen using the rectifier activation function
# better performance is seen using the rectifier activation function
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
# specify the loss function to use to evaluate a set of weights
# logarithmic loss, which for a binary classification problem
# efficient gradient descent algorithm “adam” for no other reason that it is an efficient default
# because it is a classification problem, we will collect and report the classification accuracy as the metric
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
## fixed number of iterations through the dataset called epochs
## set the number of instances that are evaluated before a weight update in the network is performed called the batch size
model.fit(X, Y, nb_epoch=100, batch_size=2)

# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
