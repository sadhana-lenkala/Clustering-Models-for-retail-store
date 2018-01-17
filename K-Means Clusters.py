#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
Dataset = pd.read_csv('Mall_Customers.csv')
X = Dataset.iloc[:,[3,4]].values


#Using Elbow method to find the optimum number of clusters
from sklearn.cluster import KMeans
WCSS = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    WCSS.append(kmeans.inertia_)
plt.plot(range(1,11), WCSS)
plt.title('The Elbow Method')
plt.xlabel('No of clusters')
plt.ylabel('WCSS')
plt.show()    

#from graph as we found that 5 is ideal cluster
#use that to create k-means cluster and predit
kmeans = KMeans(n_clusters = 5, init = 'k-means++',n_init =10, max_iter = 300,random_state = 0)
y_kmeans = kmeans.fit_predict(X)

#visualizing the clusters
plt.scatter(X[y_kmeans == 0 , 0],X[y_kmeans == 0,1],color = 'red', label = 'Low spenders', s = 100)
plt.scatter(X[y_kmeans == 1 , 0],X[y_kmeans == 1,1],color = 'blue', label = 'standard', s = 100)
plt.scatter(X[y_kmeans == 2 , 0],X[y_kmeans == 2,1],color = 'green', label = 'Target', s = 100)
plt.scatter(X[y_kmeans == 3 , 0],X[y_kmeans == 3,1],color = 'magenta', label = 'Low earners', s = 100)
plt.scatter(X[y_kmeans == 4 , 0],X[y_kmeans == 4,1],color = 'cyan', label = 'out of target', s = 100)
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s = 300, c= 'yellow', label = 'cluster centers')
plt.title('cluster of clients')
plt.xlabel('Annual income (k$)')
plt.ylabel('Spending score (1-100)')
plt.legend()
plt.show()
