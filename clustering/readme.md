## Clustering
**Clustering** is one of the most popular unsupervised learning technique(where we have input data but no correspondin output )<br>
It can be defined as identifying similar characterstics of data in a give dataset

**K-Means Clustering:**<br>
One of the most popular technique for clustering is K-Means<br>

**K-Means Clustering Working**<br>
Two Steps<br>
**1.Cluster Assignment**: Assign data points to each centroid depending on how close the data points are to centroids using some similarity measures such as 
   **Euclidean Distance** = d =square_root(summation(Xi-Yi))^2 where i is 1...N<br>
**2.Move Centroids**: It moves Centroids by **calculating Mean**<br>
**Repeat until there is no change in centroid values**
    
**Identifying Optimal number of clusters:** <br>
**1. Elbow Method** <br>
**2. Silhoutte Co-efficient** <br>

**Silhoutte Co-efficient** Silhouette values lies in the range of [-1, 1].<br>
A value of +1 indicates that the sample is far away from its neighboring cluster and very close to the cluster its assigned.<br>
Similarly, value of -1 indicates that the point is close to its neighboring cluster than to the cluster its assigned and, a value of 0 means its at the boundary of the distance between the two cluster. 
Value of +1 is ideal and -1 is least preferred.<br>
Hence, higher the value better is the cluster configuration<br>

**one should consider the following points to identify good number of clusters**<br>
1.**Firstly**, The mean value should be as close to 1 as possible<br>
2.**Secondly**, The plot of each cluster should be above the mean value as much as possible. Any plot region below the mean value is not desirable.<br>
3.**Lastly**, the width of the plot should be as uniform as possible
