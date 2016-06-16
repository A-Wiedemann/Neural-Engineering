


__author__ = 'Nicolas'

import gzip
import numpy
import time
from sklearn import svm, grid_search
import cPickle
start_time = time.clock()

def unpickle(file):
    #import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

# Unpack all batches
dict1 = unpickle('data_batch_1')
data1 = dict1['data']
labels1 = dict1['labels']

dict2 = unpickle('data_batch_2')
data2 = dict2['data']
labels2 = dict2['labels']

dict3 = unpickle('data_batch_3')
data3 = dict3['data']
labels3 = dict3['labels']

dict4 = unpickle('data_batch_4')
data4 = dict4['data']
labels4 = dict4['labels']

dict5 = unpickle('data_batch_5')
data5 = dict5['data']
labels5 = dict5['labels']

# put all dataset_batches together to form the whole dataset
data = numpy.concatenate((data1,data2,data3,data4,data5))
labels = numpy.concatenate((labels1,labels2,labels3,labels4,labels5))

print('Data size: ', numpy.shape(data))
print('Labels size: ', numpy.shape(labels))

# unpack the test dataset

test = unpickle('test_batch')
test_data = test['data']
test_labels = test['labels']

###u### only needed for submission: unpacking of the 300.000 test data ###################
'''
file_recover_testmatrix = open('testmatrix','r')
test_data = cPickle.load(file_recover_testmatrix)
file_recover_testmatrix.close()
'''
print('Test data size: ', numpy.shape(test_data))
#print('Test labels size: ', numpy.shape(test_labels))

#print(data)
#print(labels)

X = data[0:5000]
y = labels[0:5000]

'''
########################## Do ZCA Whitening with the data ##############################
from sklearn.base import BaseEstimator, TransformerMixin
class ZCA(BaseEstimator, TransformerMixin):
    """
    Identical to CovZCA up to scaling due to lack of division by n_samples
    S ** 2 / n_samples should correct this but components_ come out different
    though transformed examples are identical.
    """
    def __init__(self, n_components=None, bias=.1, copy=True):
        self.n_components = n_components
        self.bias = bias
        self.copy = copy


    def fit(self, X, y=None):
        if self.copy:
            X = numpy.array(X, copy=self.copy)
        n_samples, n_features = X.shape
        self.mean_ = numpy.mean(X, axis=0)
        X -= self.mean_
        U, S, VT = numpy.linalg.svd(X, full_matrices=False)
        components = numpy.dot(VT.T * numpy.sqrt(1.0 / (S ** 2 + self.bias)), VT)
        self.components_ = components[:self.n_components]
        return self

    def transform(self, X):
        if self.copy:
            X = numpy.array(X, copy=self.copy)
            X = numpy.copy(X)
        X -= self.mean_
        X_transformed = numpy.dot(X, self.components_.T)
        return X_transformed

zca = ZCA()
#X = zca.fit_transform(X)
#test_data=zca.fit_transform(test_data)
###############################################################################################
'''

print numpy.shape(X)
#param_grid = {'kernel': ('linear', 'rbf'), 'C': [0.01, 10], 'gamma': [0.0001, 0.1]}

############################# construct classifier and fit the data ##########################
clf = svm.SVC(kernel="linear", C=0.01, gamma=0.001)
#svr = svm.SVC()
#clf = grid_search.GridSearchCV(svr,param_grid)

clf.fit(X,y)
print("Prediction for first test image: ")
print(clf.predict(test_data[0:15]))
#print("The image's actual label is: ")
#print(test_labels[0:15])


pred = clf.predict(test_data)
print('Size of pred: ', numpy.shape(pred))

'''
############### Write the resulting predictions in a csv file #####################
import csv
with open('predictions.csv','wb') as csvfile:
    result_writer = csv.writer(csvfile,delimiter=',')
    result_writer.writerow(['id','label'])

############### Find the names of the predictions ################################
    meta = unpickle('batches.meta')
    #print(meta)
    label_names = meta['label_names']
    output = numpy.zeros(300000,dtype=object)
    for i in range(0,300000):
        output[i] = label_names[pred[i]]
        result_writer.writerow([i+1,output[i]])

print'Predictions: ', output[0:5]
'''


from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, test_labels)
print("accuracy: ", acc)


#print(clf.best_estimator_.get_params())
#print(clf.best_params_)



end_time = time.clock()
print('Elapsed Time: ', end_time - start_time)
