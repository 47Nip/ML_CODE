import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer  # ✅ correct import

# Step 1: Load Breast Cancer dataset
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['diagnosis'] = cancer.target  # ✅ updated target column name

# Step 2: Encode the target variable
# Target is already numeric (0 = malignant, 1 = benign)

# Step 3: Features and target
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Step 4: Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=5
)

# Step 5: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

# Step 7: Predictions
y_pred = knn.predict(X_test_scaled)

# Step 8: Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Step 9: Show predictions
test_set_with_predictions = X_test.copy()
test_set_with_predictions['Actual'] = y_test.values
test_set_with_predictions['Predicted'] = y_pred
print(test_set_with_predictions)
