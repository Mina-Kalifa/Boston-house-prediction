#!/usr/bin/env python
# coding: utf-8

# In[458]:


from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2 , f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


# In[459]:


# Load the dataset
boston = load_boston()

# Convert to pandas DataFrame
data = pd.DataFrame(boston.data, columns=boston.feature_names)
# Add the target variable to the dataframe
data['price'] = boston.target


# In[460]:


y = pd.DataFrame(boston.target)


# In[461]:


data.head()


# In[462]:


data.describe()


# In[463]:


data.isnull().sum()


# In[464]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(data.corr(), vmin=-1, vmax=1, cmap="rocket_r")
plt.show()


# In[465]:


corr_matrix = data.corr()

# Print the correlation matrix
print(corr_matrix)


# In[466]:


data = data.drop(columns=['price'])


# In[467]:


scaler = StandardScaler()
scaler.fit(data)
X_new = scaler.transform(data)


# In[468]:



# Perform feature selection using SelectKBest with f_regression score function
k_best = SelectKBest(score_func=f_regression, k=6) # Select top 6 features
X_new = k_best.fit_transform(X_new, y)


# In[ ]:





# In[469]:


X_train, X_test, y_train, y_test = train_test_split(
    X_new, y, test_size=0.33, random_state=1)


# In[470]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)
X_ptest = poly.transform(X_test)
# Create a Linear Regression object
lr = LinearRegression()

# Train the model on the training data
lr.fit(X_poly, y_train)

# Predict the target variable on the test data
y_pred_lr = lr.predict(X_ptest)

# Evaluate the model's performance on the test data
mse_lr = mean_squared_error(y_test, y_pred_lr)
print("Linear Regression MSE:", mse_lr)
r2 = r2_score(y_test, y_pred_lr)
print("R2 score:", r2)


# In[ ]:





# In[471]:


from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Create a Ridge Regression object
ridge = Ridge(alpha=0.5)

# Train the model on the training data
ridge.fit(X_train, y_train)

# Predict the target variable on the test data
y_pred_ridge = ridge.predict(X_test)

# Evaluate the model's performance on the test data
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
print("Ridge Regression MSE:", mse_ridge)
r2 = r2_score(y_test, y_pred_ridge)
print("R2 score:", r2)


# In[472]:


from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

# Create a Lasso Regression object
lasso = Lasso(alpha=0.1)

# Train the model on the training data
lasso.fit(X_train, y_train)

# Predict the target variable on the test data
y_pred_lasso = lasso.predict(X_test)

# Evaluate the model's performance on the test data
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
print("Lasso Regression MSE:", mse_lasso)
r2 = r2_score(y_test, y_pred_lasso)
print("R2 score:", r2)


# In[473]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Create a Random Forest Regression object
rf = RandomForestRegressor(n_estimators=100)

# Train the model on the training data
rf.fit(X_train, y_train)

# Predict the target variable on the test data
y_pred_rf = rf.predict(X_test)

# Evaluate the model's performance on the test data
mse_rf = mean_squared_error(y_test, y_pred_rf)
print("Random Forest Regression MSE:", mse_rf)
r2 = r2_score(y_test, y_pred_rf)
print("R2 score:", r2)


# In[474]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Create a Gradient Boosting Regression object
gb = GradientBoostingRegressor(n_estimators=100)

# Train the model on the training data
gb.fit(X_train, y_train)

# Predict the target variable on the test data
y_pred_gb = gb.predict(X_test)

# Evaluate the model's performance on the test data
mse_gb = mean_squared_error(y_test, y_pred_gb)
print("Gradient Boosting Regression MSE:", mse_gb)
r2 = r2_score(y_test, y_pred_gb)
print("R2 score:", r2)


# In[ ]:





# In[475]:


# Define the model
svr = SVR()

# Define the hyperparameters to tune
params = {'kernel': ['rbf'],
          'C': [100],
          'gamma': ['scale']}

# Define the GridSearchCV object
grid_svr = GridSearchCV(svr, param_grid=params, cv=5, n_jobs=-1)

# Fit the GridSearchCV object to the training data
grid_svr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = grid_svr.predict(X_test)

# Evaluate the model
print("MSE of SVR : ", mean_squared_error(y_test, y_pred))
print("RMSE of SVR : ", np.sqrt(mean_squared_error(y_test, y_pred)))
print(f"R^2 of SVR : {(r2_score(y_test, y_pred)):.2f}")


# In[ ]:





# In[ ]:




