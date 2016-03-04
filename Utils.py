import csv
import glob
import numpy as np
import pandas as pd
import os
import sys
import datetime as dt

def read_price_from_csv(key, data_point, i,bottom=True):
    prices = []
    filename = os.path.join(os.path.expanduser('~'), 'RA', 'MachineLearningProject','data','csv', key +'*.csv')
    filenames = glob.glob(filename)

    if len(filenames) >= 1:
        filename = filenames[0]
        with open(filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
            rows = list(reader)
            if bottom:
                start = -25000+1000*i
                prices = np.array(rows[start:start+12500])
            else:
                prices = np.array(rows[0:data_point])
            prices = np.transpose(prices)

    return prices

def read_signals_from_file(key, data_point, i):
    
    filename = os.path.join(os.path.expanduser('~'), 'RA', 'MachineLearningProject', 'data','random_forest',key +'_'+str(i)+'*')
    filenames = glob.glob(filename)

    if len(filenames) >= 1:
        filename = filenames[0]
        with open(filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
            signals = np.array(list(reader))[:,1:]
    return signals
    
def convert_to_time(timestamp):
    temp = dt.timedelta(timestamp)
    time = dt.datetime(0001, 1, 1) + dt.timedelta(days=temp.days) + dt.timedelta(seconds=temp.seconds)
    # time = tm.gmtime(dt.timedelta(timestamp).seconds)
    #return time.strftime("%Y/%m/%d %H:%M:%S.%f")
    return time
    

if __name__ == '__main__':
    print(read_price_from_csv('NG', 12500,1))
    print(read_signals_from_file('NG', 12500,1))[:10]
