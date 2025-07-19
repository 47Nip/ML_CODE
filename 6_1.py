import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the Iris dataset
iris = load_iris()

# Create DataFrame with feature data
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add target column
df['target'] = iris.target

# Define features and target
X = df.drop('target', axis=1)
y = df['target']

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=28)

# Create a Decision Tree classifier with entropy criterion
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)
s
# Make predictions
y_pred = clf.predict(X_test)

# Evaluate accuracy
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Display textual representation of the tree
print("\nDecision Tree:\n")
print(export_text(clf, feature_names=list(X.columns)))

# Visualize the decision tree
plt.figure(figsize=(12, 6))
plot_tree(clf, feature_names=X.columns, class_names=iris.target_names, filled=True,label="root")
plt.title("Decision Tree for Iris Dataset")
plt.show()
