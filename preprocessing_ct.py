# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 10:02:57 2018

@author: Christos.TSELAS , Data Analyst - Machine Learning Engineer; 
ctselas@gmail.com
"""
import pandas as pd
import math
from sklearn import preprocessing
import numpy as np
import random
random.seed(1)


class Preprocess(object):
    '''Reads the desired csv as panda DataFrame and performs preprocessing in 
        order to be ready for classification algorithms
    '''
    #Attributes:
        #directory: where the dataset is
        #name: the name of the dataset
    def __init__(self, name, directory = []):
        'Initiliaze the parameters' 
        self.directory = directory
        self.name = name
        self.train = []
        self.test = []
        self.validation = []
        self.scaler = []
        self.train_y = []
        self.test_y = []
        self.flag = 1
    
    
    def read_data(self, sample_size = 'All'):
        'Read dataset and keep samples from 0 to sample_size (integer)'
        if self.directory != []:
            if self.name[-4:] == '.csv':
                self.train = pd.read_csv(self.directory + '\\' + self.name)
            elif self.name[-5:] == '.xlsx':
                self.train = pd.read_excel(self.directory + '\\' + self.name)      
        else:
            if self.name[-4:] == '.csv':
                self.train = pd.read_csv(self.name)
            elif self.name[-5:] == '.xlsx':
                self.train = pd.read_excel(self.name)
        
        self.train = pd.DataFrame(self.train)        
        if sample_size != 'All':
            self.train = self.train[0:sample_size]


    def first_column_become_row_names(self):
        'First column become name of the rows'
        self.train.index = self.train[self.train.columns[0]] 

    def first_row_become_column_names(self):
        'First column become name of the rows'
        self.train.columns = self.train.iloc[0] 

            
    def delete_first_column(self):
        'Deletes the first column'
        self.train.drop(self.train.columns[0], axis=1, inplace=True)

    
    def delete_first_row(self):
        'Deletes the first row'
        self.train.drop(self.train.index[:1], inplace=True)
        

    def delete_constant_columns(self):
       'Deletes the columns that are constant and returns the names of them'
       constants = list(self.train.columns[self.train.iloc[0] == 
                                           np.sum(self.train,axis=0)/self.train.shape[0]])
       self.train = self.train[self.train.columns[self.train.iloc[0] != 
                                                  np.sum(self.train,axis=0)/self.train.shape[0]]]
       print ('\nThe constant columns are: ',len(constants), '\n', constants)
                
       
    def delete_constant_rows(self):
       'Deletes the rows that are constant and returns the names of them'
       constants = list(self.train[self.train[self.train.columns[0]] == 
                                   np.sum(self.train,axis=1)/self.train.shape[1]].index)
       self.train = self.train[self.train[self.train.columns[0]] != 
                               np.sum(self.train,axis=1)/self.train.shape[1]]
       print ('\nThe constant rows are: ',len(constants), '\n', constants)
       
                        
    def split_data(self, per = [0.8, 0.2], perm = 0):
        'Split dataset to train,validation and test set, randomly or not depending of "perm"' 
        # check if the input is appropriate
        if type(per) != list or len(per) > 3:
            print ('Please insert an appropriate input \
                   \nA list of at most 3 values that they sum to 1 like: \
                   \n[0.8, 0.1, 0.1], [Train, Validation, Test]')
            return 
        data_size = len(self.train)
        if perm == 1:
            self.train = self.train.sample(frac=1)
            
        if len(per) == 2:
            self.test = self.train.iloc[int(math.ceil(per[0]*data_size)):]
            self.train = self.train.iloc[:int(math.ceil(per[0]*data_size))]
        elif len(per) == 3:
            self.validation = self.train.iloc[int(math.ceil(per[0]*data_size)): 
                                        int(math.ceil((per[0]+per[1])*data_size))]
            self.test = self.train.iloc[int(math.ceil((per[0]+per[1])*data_size)):]
            self.train = self.train.iloc[:int(math.ceil(per[0]*data_size))]
        else:
            self.train = self.train.iloc[:int(math.ceil(per[0]*data_size))]
             
    def standardization(self):
        'Standarize dataset => 0 mean and 1 std'
        self.scaler = preprocessing.StandardScaler().fit(self.train[self.train.columns[:-1]])
        self.train[self.train.columns[:-1]] = self.scaler.transform(self.train[self.train.columns[:-1]])
        if type(self.test) != list:
            self.test[self.test.columns[:-1]] = self.scaler.transform(self.test[self.test.columns[:-1]])
        if type(self.validation) != list:
            self.validation[self.validation.columns[:-1]] = self.scaler.transform(self.validation[self.validation.columns[:-1]])
         
            
    def normalization(self):   
        'Normalization dataset => sum to 1'
        self.scaler = preprocessing.Normalizer().fit(self.train[self.train.columns[:-1]])
        self.train[self.train.columns[:-1]] = self.scaler.transform(self.train[self.train.columns[:-1]])
        if type(self.test) != list:
            self.test[self.test.columns[:-1]] = self.scaler.transform(self.test[self.test.columns[:-1]])
        if type(self.validation) != list:
            self.validation[self.validation.columns[:-1]] = self.scaler.transform(self.validation[self.validation.columns[:-1]])
         
            
    def scaling(self):   
        'Scale dataset to [0,1]'
        self.scaler =  preprocessing.MinMaxScaler().fit(self.train[self.train.columns[:-1]])
        self.train[self.train.columns[:-1]] = self.scaler.transform(self.train[self.train.columns[:-1]])
        if type(self.test) != list:
            self.test[self.test.columns[:-1]] = self.scaler.transform(self.test[self.test.columns[:-1]])
        if type(self.validation) != list:
            self.validation[self.validation.columns[:-1]] = self.scaler.transform(self.validation[self.validation.columns[:-1]])
               
        
    def series_to_supervised(self, n_in=1, n_out=1, dropnan=True):
        '''Convert series to supervised learning
            Using default parameters is shifting the last variable (output) 
            one step in order to be used as forecasting
        '''
        n_vars = 1 if type(self.train) is list else self.train.shape[1]
        df = pd.DataFrame(self.train)
        cols, names = list(), list()
    
    	# input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
            # forecast sequence (t, t+1, ... t+n)
            for i in range(0, n_out):
                cols.append(df.shift(-i))
                if i == 0:
                    names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
                else:
                    names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
                        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names  
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        
        agg.drop(agg.columns[list(range(df.shape[1], agg.shape[1]-1))], axis=1, inplace=True)
        self.train = agg

         
    def split_in_out(self):
         'Split into input and outputs, Using as output the final column'
         self.train, self.train_y = self.train.iloc[:, :-1], self.train.iloc[:, -1]
         if type(self.test) != list:
             self.test, self.test_y = self.test.iloc[:, :-1], self.test.iloc[:, -1]
         if type(self.validation) != list:
             self.validation, self.validation_y = self.validation.iloc[:, :-1], self.validation.iloc[:, -1]
         
    def reshape_3D(self):
         'Reshape input to be 3D [samples, timesteps, features]'
         self.train = np.array(self.train)
         self.train = self.train.reshape((self.train.shape[0], 1, self.train.shape[1]))
         if type(self.test) != list:
             self.test = np.array(self.test)
             self.test = self.test.reshape((self.test.shape[0], 1, self.test.shape[1]))
         if type(self.validation) != list:
             self.validation = np.array(self.validation)
             self.validation = self.validation.reshape((self.validation.shape[0], 1, self.validation.shape[1]))
             
             def one_hot_encodeing(idx):
                 y = np.zeros((len(idx), max(idx)+1))
                 y[np.arrange(len(idx)), idx] = 1
                 return y
         
        





