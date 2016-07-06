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
def splitList (lst, n):
    it = iter(lst)
    new = [[next(it) for _ in range(n)] for _ in range(len(lst) // n)]
    
    for i, x in enumerate(it):
        new[i].append(x)
    
    return np.asarray(new)


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
############ Setting up neural network #####################################################
x = T.TensorType('floatX', ((False,) * 5))('input')        # Notice the () at the end
y = T.ivector('targets')


X_train, y_train, X_test = utils.load_data("Data.mat")
y_train = y_train + 1                     # Don't ask

indices = []
indices.append(range(100,5440))
indices.append(range(0,100))

images = eeg_cnn_lib.gen_images(channelsLocation, X_train, 16, augment=False, pca=False, n_components=2)

# images = splitList(images, 1)             # 1 is the number of images for one label


(train,valid,test) = utils.reformatInput(images,y_train,indices)
# print(images)
    # :return:            Tensor of size [samples, colors, W, H] containing generated
    #                     images.
# def build_cnn(input_var=None, W_init=None, n_layers=(4, 2, 1), n_filters_first=32, imSize=32, n_colors=3): 
    # currently: np.shape(images) = (5440, 3, 16, 16)

n_timewin = 1

input_images = splitList(train[0],n_timewin)

# x = input_var
# y = train[1]

network = eeg_cnn_lib.build_cnn(x[0])
network = eeg_cnn_lib.build_convpool_max(x, 3)
network = eeg_cnn_lib.build_convpool_conv1d(x, 3)
network = eeg_cnn_lib.build_convpool_lstm(x, 3, n_timewin)
network = eeg_cnn_lib.build_convpool_mix(x, 3, n_timewin)
# network = eeg_cnn_lib.build_cnn(input_var=train[0], imSize=16, n_colors=3)
# network = eeg_cnn_lib.build_convpool_mix(input_vars=input_var,nb_classes=1,imSize=16,n_colors=3,n_timewin=n_timewin)
############################################################################################

############ Training the thing ############################################################
# network_output = lasagne.layers.get_output(network)

# # The loss function is calculated as the mean of the (categorical) cross-entropy between the prediction and target.
# cost = T.nnet.categorical_crossentropy(network_output,y).mean()

# # Retrieve all parameters from the network
# all_params = lasagne.layers.get_all_params(network,trainable=True)

# # Compute AdaGrad updates for training
# print("Computing updates ...")
# updates = lasagne.updates.adagrad(cost, all_params, learning_rate=0.01)

# # Theano functions for training and computing cost
# print("Compiling functions ...")
# train = theano.function([x, y], cost, updates=updates, allow_input_downcast=True)
# compute_cost = theano.function([x, y], cost, allow_input_downcast=True)






# Create a loss expression for training, i.e., a scalar objective we want
# to minimize (for our multi-class problem, it is the cross-entropy loss):
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, y)
loss = loss.mean()

# Create update expressions for training, i.e., how to modify the
# parameters at each training step. Here, we'll use Stochastic Gradient
# Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.01, momentum=0.9)


# x = T.TensorType('floatX', ((False,) * 5))()        # Notice the () at the end
# y = T.ivector('targets')
# Compile a function performing a training step on a mini-batch (by giving
# the updates dictionary) and returning the corresponding training loss:
train_fn = theano.function([x, y], loss, updates=updates)


# Finally, launch the training loop.
print("Starting training...")
# We iterate over epochs:
for epoch in range(500):
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0
    # start_time = time.time()
    for batch in iterate_minibatches(input_images, train[1], 1, shuffle=False):
        inputs, targets = batch
        train_err += train_fn(inputs, targets)
        train_batches += 1
    
    # And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1
