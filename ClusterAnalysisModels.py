##ClusterAnalysisModels.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
#K-means clustering
data = pd.read_csv('/Users/tannerbraithwaite/github/python_scripts/The Data Science Course 2020 - All Resources copy/Part_5_Advanced_Statistical_Methods_(Machine_Learning)/S38_L256/3.01. Country clusters.csv')

data_mapped = data.copy()
data_mapped['Language'] = data_mapped['Language'].map({'English':0, 'French':1, 'German':2})
x = data_mapped.iloc[:,1:4]

Kmeans = KMeans(2)
Kmeans.fit(x)
identified_clusters = Kmeans.fit_predict(x)

data_with_clusters = data.copy()
data_with_clusters['clusters'] = identified_clusters

# plt.scatter(data['Longitude'],data['Latitude'], c=data_with_clusters['clusters'],cmap='rainbow')
# plt.xlim(-180,180)
# plt.ylim(-90,90)


wcss=[]
for i in range(1,7):
    kmeans = KMeans(i)
    kmeans.fit(x)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)

number_clusters = range(1,7)

plt.plot(number_clusters, wcss)
plt.show()

##cons about kmean - you need to select a K value(the elbow method)
##sensitive to initialization(fix with k-means++)
##sensitive to outliers(remove outliers)
##produces spherical solutions
##standardization
