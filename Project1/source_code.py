
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import scale


data = pd.read_csv('traindata.csv', header=None)
labels = pd.read_csv('trainlabel.csv', header=None)
testdata = pd.read_csv('testdata.csv', header=None)

np.random.seed(2)
ind = np.random.permutation(len(data))
data = scale(data)
labels = np.array(labels)
m, n = data.shape
split = int(0.9*m)
train_data = data[ind][:split]
val_data = data[ind, :][split:]
train_labels = labels[ind][:split]
val_labels = labels[ind][split:]


# MLP Cross validation
alphas = [0.05, 0.07, 0.08, 0.1, 0.14, 0.16, 0.2, 0.3]
layers = [70, 80, 90, 100, (8, 32), (16, 64), (32, 64), (8, 16, 32)]
accuracy = []

for alpha in alphas:
    for layer in layers:
        clf = MLPClassifier(solver='lbfgs', alpha=alpha,
                            hidden_layer_sizes=layer, random_state=1)
        clf.fit(train_data, train_labels[:, 0])

        predictions = clf.predict(val_data)
        error = len(np.nonzero(predictions - val_labels[:, 0])[0])
        acc = (len(val_labels) - error) / len(val_labels)
        accuracy.append((alpha, layer, acc))
        print(alpha, layer, acc)

ind = np.array([arr[2] for arr in accuracy]).argmax()
print(accuracy[ind])


# Random forest cross validation
deps = list(range(2, 30))
best = []
for dep in deps:
    clf = RandomForestClassifier(max_depth=dep, random_state=0)
    score = np.mean(cross_val_score(clf, data, labels[:, 0], cv=10))
    best.append((dep, score))
    print(dep, score)

ind = np.array([arr[1] for arr in best]).argmax()
print(best[ind])


# Final classifier
clf = MLPClassifier(solver='lbfgs', alpha=0.07,
                            hidden_layer_sizes=70, random_state=1)
clf.fit(train_data, train_labels[:, 0])

preds = clf.predict(testdata)
preds = pd.DataFrame(preds).astype('int')
preds.columns = ['Prediction']
# preds.to_csv('project1_20450392.csv', index=False)