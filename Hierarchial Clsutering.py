#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
Dataset = pd.read_csv('Mall_Customers.csv')
X = Dataset.iloc[:,[3,4]].values

#dendogram plot to find optimum number of clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X,method ='ward',metric = 'euclidean'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distance using ward method')
plt.show()

#filtering hierarchial clustering(agglomerative) to mall dataset
from sklearn.cluster import AgglomerativeClustering
ahc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean',linkage = 'ward')
Y_ahc = ahc.fit_predict(X)

#visualizing the clusters
plt.scatter(X[Y_ahc == 0 , 0],X[Y_ahc == 0,1],color = 'red', label = 'Low spenders', s = 100)
plt.scatter(X[Y_ahc == 1 , 0],X[Y_ahc == 1,1],color = 'blue', label = 'standard', s = 100)
plt.scatter(X[Y_ahc == 2 , 0],X[Y_ahc == 2,1],color = 'green', label = 'Target', s = 100)
plt.scatter(X[Y_ahc == 3 , 0],X[Y_ahc == 3,1],color = 'magenta', label = 'Low earners', s = 100)
plt.scatter(X[Y_ahc == 4 , 0],X[Y_ahc == 4,1],color = 'cyan', label = 'out of target', s = 100)
plt.title('cluster of clients')
plt.xlabel('Annual income (k$)')
plt.ylabel('Spending score (1-100)')
plt.legend()
plt.show()
