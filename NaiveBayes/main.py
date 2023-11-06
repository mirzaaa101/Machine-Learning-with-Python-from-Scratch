from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import classification_report
from naive_bayes import GaussianNaiveBayesClassifier

X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

nb = GaussianNaiveBayesClassifier()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

cr = classification_report(y_test, y_pred)
print(cr)