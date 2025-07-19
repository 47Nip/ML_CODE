import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# Step 1: Create the dataset
data = {
    'Temp': [85, 80, 83, 70, 68, 65, 64, 72, 69, 75, 75, 72, 81, 71],
    'Humidity': [85, 90, 86, 96, 80, 70, 65, 95, 70, 80, 70, 90, 75, 91],
    'Wind Speed': [12, 9, 4, 3, 5, 20, 2, 12, 5, 2, 3, 4, 5, 15],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}
df = pd.DataFrame(data)

# Step 2: Encode the target variable
df['Play'] = df['Play'].map({'No': 0, 'Yes': 1})

# Step 3: Features and target
X = df.drop('Play', axis=1)
y = df['Play']

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
