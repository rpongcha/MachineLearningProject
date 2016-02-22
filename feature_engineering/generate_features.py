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
# Description: Given multiple time series data in (M x 2) CSV format, this script
# generates label (-1, 0, 1) and features (lagging price, moving averages, correlation)
# In the input time series data, the first column is time stamp and the second oolumn is price.
# In the current path, there are 43 symbols (43 different time series data).
# Two files per symbol are generated: *_large.bin and *_small.bin. The two files differ
# in number of datapoints that they contain.
# For lagging and moving averages, normalized price values are used.
# For calculating correlation between each symbol, return price value is used.


import pandas as pd
import glob
import numpy as np
import os
import sys
import math
import random

pd.set_option('precision', 15)

params = dict(
    path = os.path.join(os.path.expanduser('~'), 'data', 'csv', '*'),
    min_lagging = 1,
    max_lagging = 100,
    #interval_lagging = 1, #not implemented
    min_moving_average = 2,
    max_moving_average = 100,
    #interval_moving_average = 1, #not implemented
    list_epsilon = [0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001,0.00000001],
    theta = 0.001,
    max_correlation_window = 100,
    stock_count = 43,
    small_output_size = 50000,
)

#get paths to all files in 'file_path'
input_files = []
for file in glob.glob(params['path']):
    input_files.append(file)
input_files.sort()

#find the symbol with the lowest number of datapoints
#number of datapoints in the output is limited by the symbol with the lowest number of datapoints.
list_n = []
for file in input_files:
    df = pd.read_csv(file, header=None, dtype='float64')
    list_n.append(len(df))
min_n = min(list_n)
print("min_n:", min_n)

#dataframes for accumulating normalized price and return price across all symbols
df_normalized = pd.DataFrame(dtype='float64')
df_return = pd.DataFrame(dtype='float64')

for file in input_files:
    df = pd.read_csv(file, names=['Timestamp', 'Price'], header=None, dtype='float64')
    df = df.ix[:min_n]
    series_price = df.Price
    series_return = pd.Series(index = df.index, name="Return"+file, dtype='float64')

    #generate return price
    for i in range(0, min_n - 1):
        series_return[i] = (series_price[i+1]-series_price[i])/series_price[i]
    series_return = series_return.dropna()
    df_return = pd.concat([df_return, series_return], axis=1)

    #generate normalized price
    meanPrice = np.mean(series_price)
    stdPrice = np.std(series_price)

    series_normalized = pd.Series(index=series_price.index, name="PriceNormalized"+file, dtype='float64')

    for i in range(0, min_n):
        series_normalized[i] = (series_price[i]-meanPrice)/stdPrice
    df_normalized = pd.concat([df_normalized, series_normalized], axis=1)

    print("len(series_normalized)",len(series_normalized))
    print("len(series_return)", len(series_return))

for j in range(0, params['stock_count']):
    outputDataFrame = pd.DataFrame(dtype='float64')

    currNormalized = df_normalized.ix[:,j]
    currReturn = df_return.ix[:,j]
    currentFile = input_files[j]

    diffSquared = []
    #label = 1 and -1 represent increase/decrease in price. If the difference is
    #lower than epsilon, then label =0
    #In order to balance the labels as much as possible, different values of
    #epsilon are experimented and the one that balances the three classes as equally
    #as possible is chosen

    for eps in params['list_epsilon']:
        positive = 0
        neutral = 0
        negative = 0
        for i in range (0, min_n-1):
            difference = currNormalized[i+1]-currNormalized[i]
            if (difference>eps):
                positive = positive + 1
            elif (difference < (-1)*eps):
                negative = negative + 1
            else:
                neutral = neutral + 1
        total = positive + negative + neutral
        target = total / 3
        diffSquared.append((positive-target)**2+(negative-target)**2+(neutral-target)**2)
        print("epsilon:", eps)
        print("positive:", positive, positive/total)
        print("negative", negative, negative/total)
        print("neutral", neutral, neutral/total)
        print("")

    balEpsilon = params['list_epsilon'][np.argmin(diffSquared)]
    print("Selected epsilon", balEpsilon)
    print("")

    seriesLabel = pd.Series(index=currNormalized.index, name="Label"+str(balEpsilon)+currentFile, dtype='float64')
    for i in range (0, min_n-1):
        difference = currNormalized[i+1]-currNormalized[i]
        if (difference>balEpsilon):
            seriesLabel[i]=1
        elif (difference<(-1)*balEpsilon):
            seriesLabel[i]=-1
        else:
            seriesLabel[i]=0

    outputDataFrame=pd.concat([outputDataFrame, seriesLabel],axis=1)

    #generates lagging columns using normalized price,
    for i in range(1,params['max_lagging']+1):
        seriesLagged = pd.Series(currNormalized.shift(i), index=currNormalized.index, name="Lagging "+str(i)+currentFile, dtype='float64')
        outputDataFrame=pd.concat([outputDataFrame,seriesLagged],axis=1)

    #generates moving averages normalized price
    for i in range (params['min_moving_average'], params['max_moving_average']+1):
        seriesMovingAverage = currNormalized
        seriesMovingAverage = pd.rolling_mean(seriesMovingAverage, i)
        seriesMovingAverage = pd.Series(seriesMovingAverage, index=seriesMovingAverage.index, name="Moving Average"+str(i)+currentFile, dtype='float64')
        outputDataFrame = pd.concat([outputDataFrame, seriesMovingAverage], axis=1)

    #calculates correlation with different symbols using moving windows.
    #adds very small values of perturbation to avoid division by zero while
    #calculating correlation.

    for k in range (j+1, params['stock_count']):
        u = (params['theta'] * balEpsilon)/math.sqrt(params['max_correlation_window'])
        compareFile = input_files[k]

        xPrice = currReturn
        yPrice = df_return.ix[:,k]
        xTemp = pd.Series(dtype='float64')
        yTemp = pd.Series(dtype='float64')
        xTemp = xPrice.apply(lambda x: u*(random.uniform(-1,1)))
        yTemp = yPrice.apply(lambda x: u*(random.uniform(-1,1)))
        xPrice = xPrice.add(xTemp)
        yPrice = yPrice.add(yTemp)

        seriesCorrelation = pd.Series(index=outputDataFrame.index, name="Correlation"+currentFile+" VS "+compareFile, dtype='float64')

        for i in range(params['max_correlation_window'], min_n):
            correlation = np.corrcoef(xPrice[i-(params['max_correlation_window'] - 1) : i], yPrice[i-(params['max_correlation_window'] - 1) : i], bias = 1)[0][1]
            seriesCorrelation[i] = correlation

        outputDataFrame = pd.concat([outputDataFrame, seriesCorrelation], axis=1)

    #two output files are prepared
    #size of the ouput is n_min calculated initially
    #size of small output set is defined in params

    outputDataFrame = outputDataFrame.dropna()
    smallDataFrame = outputDataFrame.tail(params['small_output_size'])

    file = os.path.splitext(currentFile)[0]

    dimension = np.array([len(outputDataFrame), len(outputDataFrame.columns)])
    smallDimension = np.array([params['small_output_size'], len(outputDataFrame.columns)])

    print("dimensions for: ", currentFile)
    print("number of rows:", len(outputDataFrame))
    print("number of columns: ", len(outputDataFrame.columns))
    print("")

    #append dimension (n_row, n_column) to beginning of file and export to binary
    outputArray = outputDataFrame.as_matrix()
    outputArray=np.append(dimension,outputArray)
    outputArray.astype('float64')
    outputArray.tofile(file+'_large.bin')
    smallOutputArray = smallDataFrame.as_matrix()
    smallOutputArray=np.append(smallDimension,smallOutputArray)
    smallOutputArray.astype('float64')
    smallOutputArray.tofile(file+'_small.bin')

    #for outputting to csv format
    # outputDataFrame.to_csv(file+'_largeHybrid.csv',index=False)
    # smallDataFrame.to_csv(file+'_smallHybrid.csv',index=False)





