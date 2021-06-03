import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle



# Importing the dataset
dataset = pd.read_csv('t20_matches.csv')
'''dataset['innings1_runs'].fillna(0,inplace=True)
dataset['innings1_wickets'].fillna(0,inplace=True)
dataset['innings1_overs_batted'].fillna(0,inplace=True)
dataset['innings2_runs'].fillna(0,inplace=True)
dataset['target'].fillna(0,inplace=True)
print(dataset.isnull().any())
'''
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

X1 = dataset.iloc[:, 15].values
X2 = dataset.iloc[:, 16].values
X3 = dataset.iloc[:, 17].values
Y = dataset.iloc[:, 20].values

X1 = X1.reshape(X1.size,1)
print(X1.size)
X1 = X1.astype(np.float64, copy=False)
print(np.isnan(X1).any())
X2 = X2.reshape(X2.size,1)
print(X2.size)
X2 = X2.astype(np.float64, copy=False)
print(np.isnan(X2).any())
X3 = X3.reshape(X3.size,1)
print(X3.size)
X3 = X3.astype(np.float64, copy=False)
print(np.isnan(X3).any())
Y = Y.reshape(Y.size,1)
print(Y.size)
Y = Y.astype(np.float64, copy=False)
print(np.isnan(Y).any())

'''X_test = X_test.reshape(X_test.size,1)
X_test = X_test.astype(np.float64, copy=False)
print(np.isnan(X_test).any())
Y_test = Y_test.reshape(Y_test.size,1)
Y_test = Y_test.astype(np.float64, copy=False)
print(np.isnan(Y_test).any())
'''

X1[np.isnan(X1)] = dataset['innings1_runs'].mean()

Y[np.isnan(Y)] = math.floor(dataset['innings2_runs'].mean())
X2[np.isnan(X2)] = dataset['innings1_wickets'].mean()
X3[np.isnan(X3)] = dataset['innings1_overs_batted'].mean()
print(X1)
print(np.isnan(Y).any())
print(np.isnan(X3).any())

print(dataset.head(100))
#X = dataset.iloc[:, [16, 17, 18]].values

#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
X = dataset.iloc[:, [15, 16, 17]].values

# Feature Scaling
sc = StandardScaler()
X = sc.fit_transform(X)
#X_test = sc.transform(X_test)
#print(dataset.head(100))
# Fitting Random Forest Classification to the Training set

classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X,Y)
print ("Train Accuracy :: ", accuracy_score(Y, classifier.predict(X)))
pickle.dump(classifier, open("./pickles/random_forest.pickle", 'wb'))




# Fitting Naive Bayes to the Training set

classifier = GaussianNB()
classifier.fit(X, Y)
print ("Train Accuracy :: ", accuracy_score(Y, classifier.predict(X)))

pickle.dump(classifier, open("./pickles/naive_bayes.pickle", 'wb'))


# Fitting K-NN to the Training set

classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X, Y)
print ("Train Accuracy :: ", accuracy_score(Y, classifier.predict(X)))

pickle.dump(classifier, open("./pickles/knn.pickle", 'wb'))


# Fitting Decision Tree Classification to the Training set

classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X, Y)
print(classifier.predict(X))
print ("Train Accuracy :: ", accuracy_score(Y, classifier.predict(X)))

pickle.dump(classifier, open("./pickles/decision_tree.pickle", 'wb'))
