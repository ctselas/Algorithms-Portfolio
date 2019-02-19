# -*- coding: utf-8 -*-
"""
Created on Wed Sep 05 14:40:24 2018

@author: Christos.TSELAS , Data Analyst - Machine Learning Engineer; 
ctselas@gmail.com
"""
import preprocessing_ct
import visualization_ct
import classification_ct
import classification_performace_ct
import numpy as np
import random
#%%  Preprocessing
directory = r'C:\Users...'   # where the data are!     
name = 'data.csv'

# initialize class "preprocessing"    
d1 =  preprocessing_ct.Preprocess(name, directory) # preprocessing_ct.Preprocess(name) if dataset in the same folder

'read data set for a first overview'
d1.read_data() # adding an integer between 1 and the size of data we can keep a part of it in the RAM e.g. d1.read_data(sample_size = 800 )
train_X = d1.train #pd.DataFrame

#after first overview we need:
'Change names of the columns or rows using existing entities or delete cols or rows'
d1.first_column_become_row_names()
d1.delete_first_column()
#d1.first_column_become_row_names()
#d1.delete_first_column()

'delete useless entries'
d1.delete_constant_rows() 
d1.delete_constant_columns()
# In the future complete missing values, delete features without information etc.

'Split data in train and test'
d1.split_data() # default is 80% train and 20% keeping the samples in the same order! 
#split_data( per = [0.8, 0,2], perm = 0) default
#per is a list sized 1,2 or 3 as the slices that the dataset will be split (Train, Validation, Test) 
#perm is 0 or 1 which indicates shuffling or not

train_X = d1.train
test_X = d1.test

'Normalization'
d1.scaling() #Scale dataset to [0,1]
#d1.normalization() #Normalization dataset => sum to 1 
#d1.standardization() #Standarize dataset => 0 mean and 1 std

#d1.split_in_out()
#train_x = d1.train
#train_y = d1.train_y
#test_x = d1.test
#test_y = d1.test_y

#%%  Visualization
data = d1.train
data = data[0: np.where(data['labels'] == 1)[0][0] + 20]  

d2 = visualization_ct.Visualization(data, name) #flag: the default value of flag is 1 which means there is a 'target' in the last column.
d2 = visualization_ct.Visualization(data, name, flag = 0) #flag: the default value of flag is 1 which means there is a 'target' in the last column.
#%%  Classification 
d1.split_in_out()
train_x = d1.train
train_y = d1.train_y

# initiliaze class 
d2 = classification_ct.Classification(train_x, train_y, name)
'Decision Tree'
clf, meanauc = d2.decision_trees(max_depth = 5)  # 'Train and Visualize a Decision Tree'  Default max_depth = 10
# creates predictions usins CV to explore performance! And saves results as pngs and put then inside a word document

'1. Several Random Forest to find the best number of trees'
indexmaxscores, maxscores = d2.random_forest_check_trees(interval = [20, 35]) # Default interval is interval = [5, 35]

'2. One RF works like Decision trees'
importances , clf, meanauc = d2.random_forest(n_trees = 10) # Default  n_trees = 20
# creates predictions usins CV to explore performance! And saves results as pngs and put then inside a word document

'3. Find the most important features for each class'
d2.class_feature_importance(importances) #

'All three Random forest functions together'
d2.random_forests_all(interval = [20, 35])
#%%  Classification_Performance
# generate fake predicts
r_sampling = random.sample(range(1, train_y.shape[0]), 20000)
pred_y = train_y[r_sampling[0: int(len(r_sampling)/2)]]
train_y = train_y[r_sampling[int(len(r_sampling)/2):]]

# initiliaze class 
d3 = classification_performace_ct.Classification_Performance(train_y, pred_y, name) 
d3.confusionMatrix()
d3.area_under_the_ROC()

'Run all metrics together'
classification_performace_ct.Classification_Performance(train_y, pred_y, name, flag = 1) 