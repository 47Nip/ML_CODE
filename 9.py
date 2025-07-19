import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 1. Load Titanic dataset from local CSV file
df = pd.read_csv("titanic.csv")
print(df.head())

# 2. Remove target column ('survived') for unsupervised learning
df = df.drop(columns=['Survived'])

# 3. Drop irrelevant or mostly empty columns
df = df.drop(columns=['name', 'ticket', 'Cabin'], errors='ignore')

# 4. Fill missing values
df = df.ffill()


# 5. Encode any text columns into numbers
for col in df.select_dtypes(include='object'):
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# 6. Scale the features
scaler = StandardScaler()
scaled = scaler.fit_transform(df)

# 7. Use Elbow Method to find best k
inertias = []
for k in range(1, 10):
    km = KMeans(n_clusters=k, random_state=0)
    km.fit(scaled)
    inertias.append(km.inertia_)

# 8. Plot elbow curve
plt.plot(range(1, 10), inertias, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid(True)
plt.show()

# 9. Apply KMeans with chosen k (you can change based on elbow plot)
k = 3
kmeans = KMeans(n_clusters=k, random_state=0)
labels = kmeans.fit_predict(scaled)

# 10. Evaluate clustering
score = silhouette_score(scaled, labels)
print("Silhouette Score:", round(score, 4))
