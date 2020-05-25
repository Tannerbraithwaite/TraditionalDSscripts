##ClusterAnalysisModels.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
from sklearn import preprocessing
data = pd.read_csv('/Users/tannerbraithwaite/github/python_scripts/The Data Science Course 2020 - All Resources copy/Part_5_Advanced_Statistical_Methods_(Machine_Learning)/S38_L265/3.12. Example.csv')


x=data.copy()

kmeans = KMeans(2)
kmeans.fit(x)
clusters = x.copy()
clusters['clusters_pred'] = kmeans.fit_predict(x)

x_scaled=preprocessing.scale(x)
wcss=[]
for i in range(1,10):
    kmeans = KMeans(i)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

# plt.plot(range(1,10), wcss)

kmeans_new = KMeans(9)
kmeans_new.fit(x_scaled)
clusters_new = x.copy()
clusters_new['clusters_pred'] = kmeans_new.fit_predict(x_scaled)

# plt.scatter(clusters_new['Satisfaction'],clusters_new['Loyalty'], c=clusters_new['clusters_pred'], cmap='rainbow')
# plt.xlabel('Satisfaction')
# plt.ylabel('Loyalty')
# plt.show()

##producing a Dendogram
data = pd.read_csv('/Users/tannerbraithwaite/github/python_scripts/The Data Science Course 2020 - All Resources copy/Part_5_Advanced_Statistical_Methods_(Machine_Learning)/S39_L272/Country clusters standardized.csv', index_col = 'Country')
x_scaled = data.copy()
x_scaled = x_scaled.drop(['Language'], axis=1)
print(x_scaled)
sns.clustermap(x_scaled, cmap='mako')
plt.show()
