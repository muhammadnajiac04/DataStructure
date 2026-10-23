import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

iris = load_iris()
X = iris.data
print(pd.DataFrame(X,columns=iris.feature_names).head())
print("fitting model")

model = KMeans(n_clusters=3, random_state=42, n_init=10)
model.fit(X)

print("Cluster Labels:")
print(model.labels_)

print("Cluster Centers:")
print(model.cluster_centers_)