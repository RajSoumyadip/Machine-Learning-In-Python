#Created on Wed Feb 21 02:43:05 2018
#@author: soumyadipghosh
#Simple Linear Data Regresssion

# Data Preprocessing Template

#Importing The Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing DataSet

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values


#Splitting the Dataset in Training Set and Test Set

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=0)


#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""


#Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
Regressor = LinearRegression()
Regressor.fit(X_train,Y_train)


#Predicting Test Set results
Y_pred = Regressor.predict(X_test)
X_pred = Regressor.predict(X_train)

#Visualing The Training Set Results
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, Regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience(Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualing The Test Set Results
plt.scatter(X_test, Y_test, color = 'green')
plt.plot(X_train, Regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience(Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()