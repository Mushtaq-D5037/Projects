"""
@author: Mushtaq Mohammed
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# load dataset
df = pd.read_csv('turkiye-student-evaluation_generic.csv')

# some basic checks
df.columns
df.head()


# missing values
df.isnull().sum()

# value counts
for i in df.columns:
    print(df[i].value_counts())
    

# categorical and numerical features
cat = df.select_dtypes(exclude = [np.number])
num   = df.select_dtypes(include = [np.number])


# clustering data based only on Questions
df_new = df.iloc[:,5:33]

# whether to scale or not
df_new.describe()
# observation
# all the data is in the same range
# so no need of scaling

# elbow method : to find optimum number of clusters
from sklearn.cluster import KMeans

# converting to array to feed in KMeans
X = df_new.values

# with pca
from sklearn.decomposition import PCA
pca = PCA(n_components = 2, random_state=1)
X_pca = pca.fit_transform(X)


wcss = []  # within cluster sum of square

for k in range(1,11):
    k_means = KMeans(n_clusters = k,init ='k-means++',max_iter = 300,
                     n_init=10,random_state =42)
    k_means.fit(X_pca)
    wcss.append(k_means.inertia_)
    
plt.plot(range(1,11) ,wcss,marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.xticks(range(1,11))
plt.ylabel('wcss')
plt.show()

kmeans = KMeans(n_clusters=3,init='k-means++',random_state=0)
y = kmeans.fit_predict(X_pca)

plt.scatter(X_pca[y==0,0],X_pca[y==0,1],c='red',label='cluster1')
plt.scatter(X_pca[y==1,0],X_pca[y==1,1],c='blue',label='cluster2')
plt.scatter(X_pca[y==2,0],X_pca[y==2,1],c='green',label='cluster3')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=25,c='black',label='Centroid')
plt.title('kmeans clustering')
plt.X_pcalabel('X_pca')
plt.ylabel('Y')
plt.legend()
plt.show()


