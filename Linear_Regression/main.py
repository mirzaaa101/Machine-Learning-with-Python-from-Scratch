import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import datasets
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X1, y_train, y1 = train_test_split(X, y, test_size=0.3, random_state=1234)
X_val, X_test, y_val, y_test = train_test_split(X1, y1, test_size=0.5, random_state=1234)

fig = plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], y, color = "b", marker = "o", s = 30)
plt.show()

# create object of linear regression model
reg_clf = LinearRegression(lr=0.001)
reg_clf.fit(X_train, y_train)
# calculate mse for validation data
y_pred = reg_clf.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print(mse)

# mse plotting
y_pred_line = reg_clf.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
plt.show()


# calculate mse for test data
y_pred = reg_clf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(mse)
