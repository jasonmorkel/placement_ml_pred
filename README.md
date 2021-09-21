# placement_ml_pred
Predicting whether a student gets placed in a job or not based on certain variables.

# etl.py 
Student placement data is extracted from Kaggle using a kaggle API.

The data is cleaned thoroughly during this stage and prepped for model selection. 

We split the data into training and test datasets and export them to .csv files

# train.py
Support vector machines (svm), logistic regression and random forest models are applied to the training data to see which model performs the best. 

0                  svm    0.855126  {'C': 1, 'kernel': 'linear'}
1        random_forest    0.890084         {'n_estimators': 100}
2  logistic_regression    0.866723                      {'C': 1}

We see that random forests model performs the best with 89% accuracy. 

# test.py
Random forest model (n estimaters = 100) is fit to the training data using sklearn package. 

Accuracy: 0.7674418604651163

