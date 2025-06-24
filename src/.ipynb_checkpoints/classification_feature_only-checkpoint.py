import pandas as pd
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score

import shap
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('features_tianjin.csv')
columns_tabular = [i for i in df.columns if i not in ['cell', 'label', 'dataset', 'Unnamed: 0', 'is_valid', 'index', 'Unnamed: 0.1']]
X_train = df.loc[df.is_valid == False, columns_tabular]
y_train = df.loc[df.is_valid == False, ['label']]
X_test = df.loc[df.is_valid == True, columns_tabular]
y_test = df.loc[df.is_valid == True, ['label']]

clf = svm.SVC(kernel = 'linear')
clf.fit(X_train, y_train.label.tolist())

y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

def plot_coefficients(classifier, feature_names, top_features=50):
    # coef = classifier.coef_.ravel()
    coef = classifier.coef_[0]
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])

    return feature_names[top_coefficients]

# features_names = X.columns.tolist()
out = plot_coefficients(clf, np.array(columns_tabular))

X_train_reduced = df.loc[df.is_valid == False, out]
X_test_reduced = df.loc[df.is_valid == True, out]

clf.fit(X_train_reduced, y_train)
y_pred = clf.predict(X_test_reduced)

print(accuracy_score(y_test, y_pred))