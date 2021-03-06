	Author: Christos Tselas, Data Analyst - Machine Learning Engineer
	Email: christos.tselas@akka.eu ; ctselas@gmail.com
==============

Overview:
	Details about the "preprocessing_ct.py"     
    	Reads the desired csv as panda DataFrame and performs preprocessing 
		in order to be ready data analysis algorithms.
	Specifically it is created for classification problems. 
	So, it's assumed that the last collumn is the target.
    
    	Input Attributes:
        	directory: the directory where the dataset is
			name: the name of the dataset
			flag: the default value of flag is 1 which means there is a test set. 
				If flag == 0 there is no need of split the data set in train and test

==============   

Class name: 
	--> Preprocess(directory, name, flag = 1)

Internal functions:

	1. read_data(sample_size = 'All') 
        	--> Read dataset as 'pd.DataFrame' and keep samples from 0 to 'sample_size', integer. 
			The default value takes all the samples.
	   
	
	2. first_column_become_row_names()
			--> First column become name of the rows
	   

	3. first_row_become_column_names()
			--> First row become name of the columns	
	   

	4. delete_first_column
			--> Deletes the first column


	5. delete_first_row()
			--> Deletes the first row
	   

	6. delete_constant_columns()
			--> Deletes the columns that are constant and returns the names of them	   


	7. delete_constant_rows()
			--> Deletes the rows that are constant and returns the names of them
	   

	8. split_data_ordered(per)
			--> Split dataset to train and test set. 
			Keeping the samples in the same order
        		With 'per' a number in (0,1), indicates the percentage to keep for train set
			Default value for per = 0.8	   


	9. split_data( per = [0.8, 0,2], perm = 0)
			--> Split dataset to train, validation and test set.
			per is a list sized 1,2 or 3 as the slices that 
				the dataset will be split (Train, Validation, Test)  
        	perm is 0 or 1 which indicates shuffling or not
	   

	10. standardization()
			--> Standarize dataset => 0 mean and 1 std
			Not the last collumn
	   

	11. normalization()
			--> Normalization dataset => sum to 1 
			Not the last collumn dataset
	   

	12. scaling()
			--> Scale dataset to [0,1]
			Not the last collumn dataset
	   

	13. series_to_supervised(n_in=1, n_out=1, dropnan=True)
			--> Convert series to supervised learning
        		Using default parameters is shifting the last variable (output) one step in order to be used as forecasting
			'The documentation needs extension' @christos

	14. split_in_out()
			--> Split into input and outputs
        		Using as output the final column
	   

	15. reshape_3D()
			--> Reshape input to be 3D [samples, timesteps, features]
        		It is used for some special Deep Neural Networks like RNN
	   





   