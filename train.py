import pandas as pd
import numpy as np
import os
os.chdir('placement_ml_pred')
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

#load_data
x_train = pd.read_csv('x_train.csv')
y_train = pd.read_csv('y_train.csv')

#create_svm_model
model_params  ={ 
    'svm' : {
        'model' : svm.SVC(gamma = 'auto'),
        'params' : {
            'C' : [1 , 10  , 20] , 
            'kernel' : ['rbf' , 'linear']
        }
    },
    
    'random_forest' : {
        
        'model' : RandomForestClassifier(),
        'params' : {
            'n_estimators' : [1 ,5 , 10, 100]
        }
        
    },


'logistic_regression' : {
    
    'model' : LogisticRegression(solver = 'liblinear' , multi_class='auto'),
    'params' :{
        'C': [1 , 5 , 10]
    }
   }
}

scores = []
for model_name  , mp in model_params.items():
    clf = GridSearchCV(mp['model'] , mp['params'] , cv = 5 , return_train_score=False)
    clf.fit(x_train, y_train)
    scores.append({
        'model' : model_name ,
        'best_score' :clf.best_score_ ,
        'best_params' :clf.best_params_
    })

results = pd.DataFrame(scores , columns=['model' , 'best_score' , 'best_params'])
print(results)