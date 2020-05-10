
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('kc_house_data.csv')
#this function will provide the descriptive statistics of the dataset.(only int value)
dataset.describe()

import seaborn as sns
#Other methods like Back Propagation/ Forward Propagation can be used. But Correlation Matrix is best for most speedy analysis.
correlation_matrix = dataset.corr().round(2)
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)


#determine X and y variables(form correlation matrix this values ar taken as independent variables)
X = dataset.iloc[:,[4,5,11,12,19,3,13,9,17,7,8]].values
y = dataset.iloc[:, [2]].values



#converting the independent variables  with degree of polynomial 3
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=3)
X = poly_reg.fit_transform(X)
#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)


#split
from sklearn.model_selection import train_test_split
X_train ,X_test, y_train ,y_test = train_test_split(X, y, test_size=0.2, random_state = 0)



#ols
from sklearn.linear_model import SGDRegressor, LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X_train,y_train)


#predicting the value
y_pred = lin_reg.predict(X_test)
#r2 result
from sklearn.metrics import r2_score
r_squared = r2_score(y_test, y_pred)
print("Coefficient of Determination using ols method = ",r_squared)




#SGD
from sklearn.linear_model import SGDRegressor, LinearRegression
regressor = SGDRegressor(max_iter=1000, tol=1e-4, alpha =0.01, random_state = 0, eta0 = 0.0001)
regressor.fit(X_train, y_train)

#predicting the value
y_pred1 = regressor.predict(X_test)
#r2 result
from sklearn.metrics import r2_score
r_squared1 = r2_score(y_test, y_pred1)
print("Coefficient of Determination using sgd method= ",r_squared1)



