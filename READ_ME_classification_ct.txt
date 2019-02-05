	Author: Christos Tselas, Data Analyst - Machine Learning Engineer
	Email: christos.tselas@akka.eu ; ctselas@gmail.com
==============

Overview:
	Details about the "classification_ct.py"     
    	Given a dataset (in a pd.DataFrame format) train some basic classification models
		(Decision Trees, Random Forests etc.) to have an overview of the classification
		performance of the dataset, linearity or nonlinearity of the problem, feature importances etc..
		Creates also a word document - Report with all the results
    
    	Input Attributes:
        	train_x: a 'pd.DataFrame' with rows the samples and columns the features
			train_y: a 'pd.DataFrame' with rows the samples and a column the target
			test_x: a 'pd.DataFrame' with rows the samples and columns the features
				default = [] (the models are trained using Cross-Validation)
			test_y: a 'pd.DataFrame' with rows the samples and a column the target
				default = [] (the models are trained using Cross-Validation)

==============   

Class name: 
	--> Classification(train_x, train_y, test_x = [], test_y = [])

Internal functions:

	1. decision_trees(max_depth = 10) 
        	--> Train and Visualize a Decision Tree
			max_depth = the maximum depth of the tree
	   
	
	2. random_forests_all(interval = [5, 35])
			--> Perform 3 different experiments using RF
				calls internal functions:
					*random_forest_check_trees(interval)#(inf:3)checks the best size (number of trees) of RF 
					*random_forest(n_trees)#(inf:4)finds the most important features
					*class_feature_importance(importances)#(inf:6)find the most important features for each class
				interval = the size of the forests that we train (number of trees)

	
	3. random_forest_check_trees(interval = [5, 35])
			--> Train several RF using 10-fold CV and find the proper number of trees
				check the best size (number of trees) of RF inside interval
				
				output: best value for number of trees and its AUC
	   

	4. random_forest(n_trees = 20)
			--> Train a RF with all data and reveal the most important features
			Prints, plots and saves the features ordered with their importances
			n_trees = size of the forest


	5. class_feature_importance_help(feature_importances)
			--> To get the importance according to each class
			Auxiliary function for class_feature_importance(importances)(inf:6)
	   

	6. class_feature_importance(importances)
			--> Plots and saves the feature importances of the forest for each class
			Uses class_feature_importance_help(inf:5)


	