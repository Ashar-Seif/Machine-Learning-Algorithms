from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import seaborn as sns
import pandas as pd
import numpy as np
import random


# Creating a random numbers list for random seed in train_test_split 
randomlist = []
for i in range(0,10):
   random_seed = random.randint(1,50) 
   randomlist.append(random_seed)


# Reading data in dataframe
Data  = pd.read_csv(r"Task_data\data.txt", delimiter = " ")
X=Data.iloc[:, :-1]
y=Data.iloc[:, -1]
print(X.shape)   #(799, 24)
print(y.shape)   #(799,)


# Preprocessing
# 1. check for missing values
print(Data.isna().any().any())

# 2. Examining distribution of target column
target=y
#print(target.unique())
#The target column has two values:

    ## 1: representing a good loan
    ## 2: representing a bad (defaulted) loan.
#The usual convention is to use '1' for bad loans and '0' for good loans. Let's replace the values to comply to the convention.
le= LabelEncoder()
le.fit(Data.iloc[:,-1])
Data.iloc[:,-1]=le.transform(Data.iloc[:,-1])
#print(Data.iloc[:,-1].head(100))


# 3. Standerization of features 
# scale the data
#scaler = StandardScaler()
#X = scaler.fit_transform(X)
#X=np.array(X)





# Accuracy lists 
accuracy=[]
accuracy_norm=[]
accuracy_train_norm=[]


# Iterating 10 times with different random seed  to create different train and test data 
for n_trial in range(len(randomlist)):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=randomlist[n_trial])    

    # Normalization : The formular is: x_norm = (x - x_min) / (x_max - x_min)
    # 
    train_min = X_train.min()
    train_range = (X_train - train_min).max()
    X_train_norm = (X_train - train_min)/train_range

    test_min = X_test.min()
    test_range = (X_test - test_min).max()
    X_test_norm= (X_test - test_min )/ test_range
    X_test_train_norm = (X_test - train_min)/ train_range
    
    
    #Lists for different experiements 
    X_train_list=[X_train,X_train_norm,X_train_norm]
    X_test_list=[X_test,X_test_norm,X_test_train_norm]
    acc=[accuracy,accuracy_norm,accuracy_train_norm]


    #Different experiements loop 
    for exp in range(len(X_train_list)):

          # train the model
          svc_model = SVC(kernel='linear')
          y_pred = svc_model.fit(X_train_list[exp], y_train).predict(X_test_list[exp])
          acc[exp].append(accuracy_score(y_test, y_pred))


 
print("The mean accuracy of data =",np.mean(accuracy))
print("The mean accuracy of normalized data =",np.mean(accuracy_norm))
print("The mean accuracy of normalized data with train parameters =",np.mean(accuracy_train_norm))



