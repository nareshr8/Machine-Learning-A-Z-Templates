# K Means Clustering
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

# Calculate the Elbow method value
from sklearn.cluster import KMeans
wCss =[]
for i in range(1,11):
    kmeans =KMeans(n_clusters= i, init = 'k-means++'
                   ,n_init= 10 , max_iter= 300, random_state=0)
    kmeans.fit(X)
    wCss.append(kmeans.inertia_)

plt.plot(range(1,11),wCss)
plt.title('K-Means Elbow Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('wCss')
plt.show


# Now Create the K Means cluster with clusters as 5
kmeans = KMeans(n_clusters=5 ,init = 'k-means++', n_init =10, 
                max_iter=300, random_state =0)
y_kmeans = kmeans.fit_predict(X)

# Plot the Cluster in a graph
plt.scatter(X[y_kmeans == 0,0],X[y_kmeans == 0,1], s = 100, c='red', label = 'Group 1')
plt.scatter(X[y_kmeans == 1,0],X[y_kmeans == 1,1], s = 100, c='green', label = 'Group 2')
plt.scatter(X[y_kmeans == 2,0],X[y_kmeans == 2,1], s = 100, c='blue', label = 'Group 3')
plt.scatter(X[y_kmeans == 3,0],X[y_kmeans == 3,1], s = 100, c='cyan', label = 'Group 4')
plt.scatter(X[y_kmeans == 4,0],X[y_kmeans == 4,1], s = 100, c='magenta', label = 'Group 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s = 300, c='yellow', label = 'Centroids')
plt.show()
