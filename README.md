# Algorithms-Portfolio

Implementations of several important algorithms for Data Analysis and Machine Learning

1. welfords_algorithm.py\
    Incremental calculation of mean and variance. Necessary for online training of machine learning methods.\
 
2. preprocessing_ct.py\
    Reads the desired csv or xlsx as panda DataFrame and performs some preprocessing 
		in order the data to be ready for data analysis algorithms. (more details: READ_ME_preprocessing_ct.txt)\
        
3. visualization_ct.py\
    Given a dataset (in a pd.DataFrame format) creates initial visualizations of the data to obtain the first 
		opinion about the data's structure, patterns, relationships etc. (more details: READ_ME_visualization_ct.txt)\
        
4. classification_ct.py\
    Given a dataset (in a pd.DataFrame format) train some basic classification models
		(Decision Trees, Random Forests etc.) to have an overview of the classification
		performance of the dataset, linearity or nonlinearity of the problem, feature importances etc..
		Creates also a word document - Report with all the results. (more details: READ_ME_classification_ct.txt)\
        
5. classification_performace_ct.py\
    Given two vectors one with actual classes and one with the predicted, it calculates and visualizes performance metrics like AUC,           Confusion Matrix etc.\
