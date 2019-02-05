	Author: Christos Tselas, Data Analyst - Machine Learning Engineer
	Email: christos.tselas@akka.eu ; ctselas@gmail.com
==============

Overview:
	Details about the "visualization_ct.py"     
		Given a dataset (in a pd.DataFrame format) creates initial visualizations of the data to obtain the first 
		opinion about the data's structure, patterns, relationships etc.
    
    	Input Attributes:
        	data: a 'pd.DataFrame' with rows the samples and columns the features
			name: name of the dataset
			flag: the default value of flag is 1 which means there is a 'target' in the last column. 
				If flag == 0 there is no target
				

==============   

Class name: 
	--> Visualization(data, name, flag = 1)
		Takes as input 'data' in a format of 'pd.DataFrame', can be obtained giving the 'directory' and the 'name' in 'preprocessing_ct' as follows:
			d = preprocessing_ct.Preprocess(directory, name) 
			data = d.data
			
		Then there is an interactive question which is:
		Hi! We have these 5 different options for visualization:
                           1. Principal Component Analysis, and visualization of first 2 PCs
                           2. Visualization of each feature and relation with target
                           3. Plot of correlation matrix
                           4. Plot the distribution of each feature
                           0. All the above(and some extra details) in a .doc file
                           Choose one of the options pressing its number
						   
						   
			

			
Internal functions:

	1. pca_vis() 
        	--> Principal Component Analysis, and visualization of first 2 PCs
			If flag == 1: use different colors for each target otherwise use the same color
	   
	
	2. feature_vis()
			--> Visualization of each feature (x-axis samples, y-axis value of the feature)
			If flag == 1: the target is also visualized with different color
	   

	3. correlation_matrix()
			--> Plot of correlation matrix
			
	   
	   
