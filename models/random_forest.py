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
# Description: This file uses Random Forest Classifier from sklearn to
# train and make predictions. load_data function should be modified.
# load_data should output two matrices x and y. x is (M x N) features matrix
# y is (M x S), where M is number of data points, N is number of features
# and S is number of symbols. Also, y can be multi-class.
# Also, by setting feature_reduction in params, one can optionally conduct
# PCA on the features set.
# The script provides two metrics: f1-score and classification error.

import glob
import pandas as pd
import numpy as np
import math
import time
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import csv
import os
import pandas as pd
#np.set_printoptions(edgeitems=30)

params = dict(
#path = os.path.join(os.path.expanduser('~'), 'RA','MachineLearningProject', 'data', 'smallBinaryPrice',asset_symbol+'*'), 
n_row = 50000,
frac_train = 0.5, # fraction of dataset used for train. 1 - frac_train is used for test.
n_symbol = 1,
feature_reduction = 0, # No. features after PCA. Change this value to an int value to conduct PCA on feature set.
n_estimator = 100, # No. estimators for RandomForestClassifier
criterion = 'entropy'
)

def load_data(file_path):
    '''
    Preprocess current dataset, which is split into several files in .bin format.

    :param file_path: path to the dataset
    :return:
    '''

    #get paths to all files in 'file_path'
    print(file_path)
    files = []
    for file in glob.glob(file_path):
        files.append(file)
        print(file)
    files.sort()

    #dataframe for appending labels and features from all .bin files
    #pandas is used because numpy ndarrays need to be initialized to their final size.
    dfLabel = pd.DataFrame(dtype="float64")
    dfFeature = pd.DataFrame(dtype="float64")

    for file in files:
        #The first two entries of the .bin file are number of rows and number of columns, respectively
        binary = np.fromfile(file, dtype='float64')
        numRow = binary[0]
        numCol= binary[1]
        binary = np.delete(binary, [0, 1])
        binary = binary.reshape((numRow, numCol))

        #concatenate all label and features
        tempLabel = pd.DataFrame(binary[:,0])
        tempFeature = pd.DataFrame(binary[:,1:])
        dfLabel = pd.concat([dfLabel, tempLabel], axis=1)
        dfFeature = pd.concat([dfFeature, tempFeature], axis=1)

        #reduce number of rows to match params['n_row']
        dfLabel = dfLabel.tail(params['n_row'])
        dfFeature = dfFeature.tail(params['n_row'])
        y = dfLabel.as_matrix()
        x = dfFeature.as_matrix()

    print("DIMENSIONS")
    print("x", x.shape)
    print("y", y.shape)
    return x, y

def train_test_split(x, y, i):
    '''
    split x and y into x_train, x_test, y_train, y_test

    :param x: numpy ndarray
    :param y: numpy ndarray
    :return: x_train, x_test, y_train, y_test
    '''
    
    startindex=1000*i
    splitIndex=1000*i+math.floor(params['frac_train']*params['n_row'])
    endindex = splitIndex+math.floor(params['frac_train']*params['n_row']*0.5)
    y_test = y[splitIndex:endindex]
    y_train = y[startindex:splitIndex]
    x_test = x[splitIndex:endindex]
    x_train = x[startindex:splitIndex]

    print("DIMENSIONS")
    print("x_test", x_test.shape)
    print("x_train", x_train.shape)
    print("y_test",y_test.shape)
    print("y_train", y_train.shape)

    return x_train, x_test, y_train, y_test

def pca(x):
    '''
    :param x: numpy ndarray
    :return: transformed x. numpy ndarray
    '''
    #pca = PCA(n_components=params['feature_reduction'])
    pca = PCA()
    x = pca.fit_transform(x)

    return x

def random_forest(x_train, x_test, y_train, y_test):
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

def print_f1_score(y_test, y_pred):
    y_pred = y_pred.ravel()
    y_test = y_test.ravel()

    #Total f1score
    macro_f1= f1_score(y_test, y_pred, average='macro')
    micro_f1= f1_score(y_test, y_pred, average='micro')
    weighted_f1=f1_score(y_test, y_pred, average='weighted')
    print("macro", macro_f1)
    print("micro", micro_f1)
    print("weighted", weighted_f1)
    score1 = pd.DataFrame({'macro':[macro_f1],'micro':[micro_f1],'weighted':[weighted_f1]})
    
    class_report = classification_report(y_test, y_pred)
    print(class_report)
    report_list=class_report.splitlines()
    report_list[-1]=report_list[-1][:3]+report_list[-1][4]+report_list[-1][6:]
    report_list=[row.split() for row in report_list]
    score2 = pd.DataFrame({'-1':[float(report_list[2][3])],'0':[float(report_list[3][3])],'1':[float(report_list[4][3])],'avg/total':[float(report_list[6][3])]})
    return score1, score2

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
    print("correct/total",float(correct) / total)
    return pd.DataFrame({"correct": [correct],"total": [total],'correct/total':[float(correct) / total]})

def process_machine_learning(symbol, i):
    params['path']=os.path.join(os.path.expanduser('~'), 'RA','MachineLearningProject', 'data', 'smallBinaryPrice',symbol+'*')
    
    x,y = load_data(params['path'])
    if params['feature_reduction']:
        x = pca(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, i)
    y_pred = random_forest(x_train, x_test, y_train, y_test)

    signal_pd = pd.DataFrame({'y_test':y_test[:,0],'y_pred':y_pred})
    signal_pd.to_csv(os.path.join(os.path.expanduser('~'), 'RA','MachineLearningProject', 'data', 'random_forest',symbol+'_'+str(i)+'.csv'),header = False)

    
if __name__ == "__main__":
    
    #log = open('../../log/pca_rf', 'w')
    #sys.stdout = log
    signals, report = process_machine_learning('NG',10)
    print report
   
    #log.close()
