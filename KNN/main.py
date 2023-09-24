from KNN import KNN
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['red','green','blue'])

# load iris dataset
iris = datasets.load_iris()

X, y = iris.data, iris.target

# Create train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# sctter plot for visualize
plt.figure()
plt.scatter(X[:,2], X[:,3], c=y, cmap=cmap, edgecolors='k', s=20)
plt.show()

# create classifier of scratch KNN
clf = KNN(k=15)
clf.fit(X_train,y_train)
y_predicted = clf.predict(X_test)

cr = classification_report(y_test, y_predicted)
print(cr)