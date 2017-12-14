import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix
b_dir = "../creditcardfraud/"

raw = pd.read_csv(b_dir + 'creditcard.csv')
X = raw.ix[:, 1:29]
Y = raw.Class
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
# Using undersampling

# One final way to improve models with an unbalanced dataset like this is to use undersampling.
# This means training the model on a training set where the “normal” data is undersampled so it has the same size as the fraudulent data.

fraud_records = len(raw[raw.Class == 1])
fraud_indices = raw[raw.Class == 1].index
normal_indices = raw[raw.Class == 0].index
under_sample_indices = np.random.choice(normal_indices, fraud_records, False)
raw_undersampled = raw.iloc[np.concatenate(
    [fraud_indices, under_sample_indices]), :]
X_undersampled = raw_undersampled.ix[:, 1:29]
Y_undersampled = raw_undersampled.Class
X_undersampled_train, X_undersampled_test, Y_undersampled_train, Y_undersampled_test = train_test_split(
    X_undersampled, Y_undersampled, test_size=0.3)
tree_classifier = tree.DecisionTreeClassifier()
tree_classifier.fit(X_undersampled_train, Y_undersampled_train)

Y_full_pred = tree_classifier.predict(X_test)
cnf_matrix = confusion_matrix(Y_test, Y_full_pred)
print(cnf_matrix)
#[[77255  8034]
#[    3   151]]
print('Accuracy: ' + str(np.round(100 * float((cnf_matrix[0][0] + cnf_matrix[1][1])) / float(
    (cnf_matrix[0][0] + cnf_matrix[1][1] + cnf_matrix[1][0] + cnf_matrix[0][1])), 2)) + '%')
print('Recall: ' + str(np.round(100 *
                                float((cnf_matrix[1][1])) / float((cnf_matrix[1][0] + cnf_matrix[1][1])), 2)) + '%')
# Accuracy: 90.59%
# Recall: 98.05%

with open("tree_classifier.dot", "w") as f:
    f = tree.export_graphviz(tree_classifier, out_file=f)
import os
os.system('dot -Tpng tree_classifier.dot -o tree_classifier.png')
