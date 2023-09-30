import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection as model_selection
from sklearn.datasets import load_wine
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import classification_report

wine = load_wine()

X = wine.data[:, :3]
y = wine.target

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.70, test_size=0.30, random_state=0)

linear_reg = LinearRegression().fit(X_train, y_train)

y_pred = linear_reg.predict(X_test)
y_pred = [int(round(val)) for val in y_pred]
names = [str(digit) for digit in wine.target_names]

accuracy = accuracy_score(y_test, y_pred)

print('Accuracy (Linear Regression): ', "%.2f" % (accuracy * 100))

print(classification_report(y_test, y_pred, target_names=names))


train_sizes = np.linspace(0.1, 0.9, 9)

accuracy_scores = []

for train_size in train_sizes:
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=train_size, test_size=1-train_size, random_state=0)
    linear_reg = LinearRegression().fit(X_train, y_train)
    y_pred = linear_reg.predict(X_test)
    y_pred = [int(round(val)) for val in y_pred]

    accuracy = accuracy_score(y_test, y_pred)

    accuracy_scores.append(accuracy)

plt.plot(train_sizes, accuracy_scores, marker='o')
plt.title('Accuracy vs. Train Size')
plt.xlabel('Train Size')
plt.ylabel('Accuracy')
plt.show()



report = classification_report(y_test, y_pred, target_names=names, output_dict=True)
precision = [report[label]['precision'] for label in names]
recall = [report[label]['recall'] for label in names]
f1_score = [report[label]['f1-score'] for label in names]

# Create a bar chart
labels = names
x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, precision, width, label='Precision')
rects2 = ax.bar(x, recall, width, label='Recall')
rects3 = ax.bar(x + width, f1_score, width, label='F1-Score')

ax.set_xlabel('Classes')
ax.set_ylabel('Scores')
ax.set_title('Classification Report Metrics')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()
