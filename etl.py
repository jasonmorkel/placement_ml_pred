from operator import le
import pandas as pd
import numpy as np
from pandas.core.tools.datetimes import Scalar
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from zipfile import ZipFile
import os

#Kaggle API
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

#import_data
api.dataset_download_file('benroshan/factors-affecting-campus-placement','Placement_Data_Full_Class.csv')
df = pd.read_csv("Placement_Data_Full_Class.csv")

#clean_data
df = df.drop(['sl_no', 'salary'] ,axis=1)

le= LabelEncoder()
df["gender_Encoded"] = le.fit_transform(df["gender"])
df["ssc_b_Encoded"] = le.fit_transform(df["ssc_b"])
df["hsc_b_Encoded"] = le.fit_transform(df["hsc_b"])
df["workex_Encoded"] = le.fit_transform(df["workex"])
df["specialisation_Encoder"] = le.fit_transform(df["specialisation"])
df["status_Encoded"] = le.fit_transform(df["status"])

onc = OneHotEncoder(handle_unknown= 'ignore' , sparse= False)
cols_encoded = pd.DataFrame(onc.fit_transform(df[["degree_t" , "hsc_s"]]))
cols_encoded.index = df.index

d= {
    0:"Comm&Mgmt",
    1:"Others",
    2:"Sci&Tech" ,
    3:"Arts", 
    4:"Commerce" ,
    5:"Science" }
cols_encoded.rename(columns =d ,inplace = True)

df_semifinal  =pd.concat([df , cols_encoded] , axis="columns")

df_semifinal.drop(['gender',
 'ssc_b',
 'hsc_b',
 'hsc_s',
 'degree_t',
 'workex',
 'specialisation',
 'status'] , axis ="columns" , inplace= True)

sc = StandardScaler()
encoded_num =pd.DataFrame(sc.fit_transform(df_semifinal[['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']]))
encoded_num.rename({
    0:"ssc_p",
    1:"hsc_p",
    2:"degree_p",
    3:"etest_p",
    4:"mba_p",
} , axis ="columns" , inplace= True)

df_final = pd.concat([encoded_num, df_semifinal[['gender_Encoded', 'ssc_b_Encoded', 'hsc_b_Encoded', 'workex_Encoded',
       'specialisation_Encoder', 'status_Encoded', 'Comm&Mgmt', 'Others',
       'Sci&Tech', 'Arts', 'Commerce', 'Science']]] ,axis ="columns")

print(df_final)

#split_the_data
y = df_final.status_Encoded
x = df_final.drop('status_Encoded', axis= "columns")

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

#save_to_csv
x_train.to_csv("x_train.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
x_test.to_csv("x_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)