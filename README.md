First Case Study for Biased Machine Learning Algorithms:

Content:
loan_data_set.csv -> initial data set created by true data generative process
loan-prediction-dream-housing-finance.ipynb -> preprocessing, visualizing loan_data_set and creating models
.joblib -> Pre trained models exported by loan-prediction-dream-housing-finance.ipynb
x_train.csv -> working data of loan-prediction-dream-housing-finance.ipynb (later used for data simulation)
FirstTry.ipynb -> Implementation of Data Simulation and Algorithm Bias Check

code/ \

preprocessing.py -> modules for preprocessing initial data
models.py -> creating machine learning models on initial data
operators.py -> implementing synthetic data generation and methods for bias assessment
