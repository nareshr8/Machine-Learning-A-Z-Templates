# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, [1]].values
y = dataset.iloc[:, 2].values

"""# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting the Model to Linear Regression
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X,y)

#print('If Linear is {}'.format(linear_regressor.predict(6.5)))

# Fitting the Model to Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
polynomial_features = PolynomialFeatures(degree = 4)
X_poly = polynomial_features.fit_transform(X)
linear_regressor2 = LinearRegression()
linear_regressor2.fit(X_poly,y)
#print('If Polynomial Linear is {}'.format(linear_regressor2.predict(6.5)))


# Visualise the result of Linear Regression
X_grid = np.arange(min(X),max(X), step =0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,marker ='X',color='red')
plt.plot(X,linear_regressor.predict(X),color='green')
plt.plot(X_grid,linear_regressor2.predict(polynomial_features.fit_transform(X_grid)),color='blue')
plt.title('Poly Linear Regressor')
plt.xlabel('Years')
plt.ylabel('Salary')
plt.show

# Predict the value using Polynomial Linear Regression
linear_regressor.predict(6.5)

# Predict the value using Polynomial Linear Regression
linear_regressor2.predict(polynomial_features.fit_transform(6.5))