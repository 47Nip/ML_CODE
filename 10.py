import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 1. Load the Breast Cancer dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

# 2. Remove the target label column
# We do NOT include data.target because this is unsupervised
# (just like we dropped 'survived' in Titanic)
# So we skip adding it to df

# 3. Check for missing values (this dataset has none, but safe to include)
df = df.ffill()  # Handles missing data just in case

# 4. Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# 5. Elbow Method to find optimal k
inertias = []
for k in range(1, 10):
    km = KMeans(n_clusters=k, random_state=0)
    km.fit(scaled_data)
    inertias.append(km.inertia_)

# 6. Plot elbow curve
plt.plot(range(1, 10), inertias, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Breast Tumour Dataset")
plt.grid(True)
plt.show()

# 7. Apply KMeans (after deciding optimal k from elbow plot)
k = 2  # this dataset has 2 actual classes (benign, malignant), so start with 2
kmeans = KMeans(n_clusters=k, random_state=0)
labels = kmeans.fit_predict(scaled_data)

# 8. Evaluate clustering
silhouette = silhouette_score(scaled_data, labels)
print("Silhouette Score:", round(silhouette, 4))

# Optional: show cluster counts
unique, counts = np.unique(labels, return_counts=True)
print("Cluster distribution:")
for i in range(len(unique)):
    print(f"Cluster {unique[i]}: {counts[i]} points")
