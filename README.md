# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 Step 1: Import the libraries and read the data frame using pandas.
Step 2: Calculate the null values present in the dataset and apply label encoder.
Step 3: Determine test and training data set and apply decison tree regression in dataset.
Step 4: calculate Mean square error,data prediction and r2. 
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Harrish Venkat V
RegisterNumber: 21222324049
*/

import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data[["Salary"]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
##MSE value


![image](https://github.com/user-attachments/assets/f64db908-f8a9-4cb6-bc4c-cd3347730f75)

## r2 Value

![image](https://github.com/user-attachments/assets/2d63dbae-ce69-4a90-a024-80499d613630)

## Data Prediction


![image](https://github.com/user-attachments/assets/10174ed8-eb94-4ffa-a45f-f80663245292)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
