from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from DecisionTrees import DecisionTree

# import dataset
data = datasets.load_breast_cancer()
X, y = data.data, data.target

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# create object of the classifier
clf = DecisionTree(max_depth=10)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

cr = classification_report(y_true=y_test, y_pred=y_pred)
print(cr)