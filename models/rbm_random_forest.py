#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#
# Author: Matthew Dixon, Diego Klabjan, Jin Hoon Bang
# Description: This file first uses RBM to learn from the given features set
# and conducts Random Forest Classification to make predictions on the provided
# dataset. load_data should be modified to load desired label (y) and feature (x).
# x is (M x N) features matrix and y is (M x S), where M is number of data points,
# N is number of features and S is number of symbols (number of label columns)
# Note that y can be multi-class.
# The script provides two metrics: f1-score and classification error.


import sys
import glob
import pandas as pd
import numpy as np
import math
import time
import csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import BernoulliRBM
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import os

# log = open('../../log/rbm_rf', 'w')
# sys.stdout = log

np.set_printoptions(edgeitems=30)

params = dict(
    path = os.path.join(os.path.expanduser('~'), 'data', 'smallHybrid', '*'),
    n_row = 50000,
    batchsize = 10,
    learning_rate = 0.001,
    n_iter = 50,
    frac_train = 0.75,
    n_symbol = 1,  # original value = 43
    reduced_feature = 500,
    n_estimator = 100,
    criterion = 'entropy'
)

print(params)

def load_data(file_path):
    files = []
    #store all files in fileOriginal[]
    for file in glob.glob(file_path):
        files.append(file)

    #dataframe for label and features
    dfLabel = pd.DataFrame(dtype="float64")
    dfFeature = pd.DataFrame(dtype="float64")
    file_name = ""

    for file in files:
        #1d array to 2d array
        binary = np.fromfile(file, dtype='float64')
        numRow = binary[0]
        numCol= binary[1]
        binary = np.delete(binary, [0, 1])
        binary = binary.reshape((numRow, numCol))
        file_name = file

        #concatenate all label and features
        tempLabel = pd.DataFrame(binary[:params['n_row'],0])
        tempFeature = pd.DataFrame(binary[:params['n_row'],1:])
        dfLabel = pd.concat([dfLabel, tempLabel], axis=1)
        dfFeature = pd.concat([dfFeature, tempFeature], axis=1)

    #reduce number of rows to match params['n_row']
    dfLabel = dfLabel.tail(params['n_row'])
    dfFeature = dfFeature.tail(params['n_row'])
    label = dfLabel.as_matrix()
    feature = dfFeature.as_matrix()

    return label, feature, file_name

def train_test_split(x, y):
    '''
    split x and y into x_train, x_test, y_train, y_test

    :param x: numpy ndarray
    :param y: numpy ndarray
    :return: x_train, x_test, y_train, y_test
    '''

    splitIndex=math.floor(params['frac_train']*params['n_row'])
    y_test = y[splitIndex:]
    y_train = y[:splitIndex]
    x_test = x[splitIndex:]
    x_train = x[:splitIndex]

    print("DIMENSIONS")
    print("x_test", x_test.shape)
    print("x_train", x_train.shape)
    print("y_test",y_test.shape)
    print("y_train", y_train.shape)

    return x_train, x_test, y_train, y_test

def print_f1_score(y_test, y_pred):
    y_pred = y_pred.ravel()
    y_test = y_test.ravel()

    #Total f1score
    print("macro", f1_score(y_test, y_pred, average='macro'))
    print("micro", f1_score(y_test, y_pred, average='micro'))
    print("weighted", f1_score(y_test, y_pred, average='weighted'))
    print(classification_report(y_test, y_pred))

def classification_error(y_test, y_pred):
    y_test = y_test.ravel()
    y_pred = y_pred.ravel()
    total = np.size(y_test)
    assert total == np.size(y_pred)
    correct = 0

    for i in range(0, total):
        if y_test[i] == y_pred[i]:
            correct += 1

    print("Classification error")
    print("correct:", correct)
    print("total:", total)
    print(correct / total)



def random_forest(x_train, x_test, y_train):
    '''

    :param x_train: numpy ndarray
    :param x_test: numpy ndarray
    :param y_train: numpy ndarray
    :param y_test: numpy ndarray
    :return: y_pred (numpy ndarray)
    '''
    start_time = time.time()

    rf = RandomForestClassifier(max_features='auto', n_estimators=params['n_estimator'], n_jobs=-1, criterion=params['criterion'])

    rf.fit(x_train, y_train)

    print('Random Forest fit time:')
    print("--- %s seconds ---" % (time.time() - start_time))

    y_pred = rf.predict(x_test)

    return y_pred


label, feature, symbol = load_data(params['path'])
pos = symbol.rfind('/')+1
symbol = symbol[pos:pos+2]

#scales values in features so that they range from 0 to 1
minmaxScaler = MinMaxScaler()
feature = minmaxScaler.fit_transform(feature)

print("Dimensions")
print("label", label.shape)
print("feature", feature.shape)

#feature selection using RBM

start_time = time.time()

rbm = BernoulliRBM(n_components=params['reduced_feature'], learning_rate=params['learning_rate'], batch_size=params['batchsize'], n_iter=params['n_iter'])
feature = rbm.fit_transform(feature)

print("RBM--- %s seconds ---" % (time.time() - start_time))

print("Dimensions after RBM")
print("label", label.shape)
print("feature", feature.shape)

x_train, x_test, y_train, y_test = train_test_split(feature, label)
y_pred = random_forest(x_train, x_test, y_train)

filename = 'PRED_'+symbol+'-5.csv'
with open(filename, 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for i in range(len(y_pred)):
        writer.writerow((y_pred[i], y_test[i][0]))

print_f1_score(y_test, y_pred)
classification_error(y_test, y_pred)

# log.close()



