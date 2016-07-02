__author__ = 'Andi'
import numpy as np
# import time
import csv
import eeg_cnn_lib
import utils

import theano
import theano.tensor as T 
import lasagne

# import scipy.io as sio



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
############################################################################################


############ Setting up neural network #####################################################
X_train, y_train = utils.load_data("Data.mat") 
images = eeg_cnn_lib.gen_images(channelsLocation, X_train, 16, augment=True, pca=True, n_components=2)

# print(images)
    # :return:            Tensor of size [samples, colors, W, H] containing generated
    #                     images.
# def build_cnn(input_var=None, W_init=None, n_layers=(4, 2, 1), n_filters_first=32, imSize=32, n_colors=3): 
    # currently: np.shape(images) = (5440, 3, 16, 16)
# input_var = T.TensorType('floatX', ((False,) * 5))()        # Notice the () at the end
target_var = T.ivector('targets')

# input_var = T.tensor4('inputs')
input_var = None
# target_var = T.ivector('targets')


# network = eeg_cnn_lib.build_cnn(input_var=input_var, imSize=np.shape(images)[2], n_colors=np.shape(images)[1])
network = eeg_cnn_lib.build_convpool_max(input_var, 3)
# l_out = lasagne.layers.DenseLayer(
#         network, num_units=1,
#         nonlinearity=lasagne.nonlinearities.softmax)
############################################################################################

############ Training the thing ############################################################


# Create a loss expression for training, i.e., a scalar objective we want
# to minimize (for our multi-class problem, it is the cross-entropy loss):
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()

# Create update expressions for training, i.e., how to modify the
# parameters at each training step. Here, we'll use Stochastic Gradient
# Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.01, momentum=0.9)



# Compile a function performing a training step on a mini-batch (by giving
# the updates dictionary) and returning the corresponding training loss:
train_fn = theano.function([input_var, target_var], loss, updates=updates)


print("Starting training...")
train_fn(X_train, y_train)
