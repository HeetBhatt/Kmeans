import random
import numpy as np
class KMeans:                  
    def __init__(self,n_clusters,iteration=1):
        self.n_clusters = n_clusters
        self.iteration = iteration
        self.centroids = None
    
    def fit_predict(self,X):
        random_index = random.sample(range(0,X.shape[0]),2)
        self.centroids = X[random_index]
        for i in range(self.iteration):
            # assign clusters
            cluster_group = self.assign_cluster(X) 
            #move centroids
            old_centroids = self.centroids
            self.centroids = self.move_centroids(X,cluster_group)
            #finish
            if(old_centroids == self.centroids).all():
                break
        return cluster_group
    
    def assign_cluster(self,X):
        distances = []
        cluster_group = []
        for row in X:
            for centroid in self.centroids:
                distances.append(np.sqrt(np.dot(row-centroid,row-centroid)))
            min_distance = min(distances)
            index = distances.index(min_distance)
            cluster_group.append(index)
            distances.clear()
        return np.array(cluster_group)
    
    def move_centroids(self,X,cluster_group):
        new_centroids = []
        for type in np.unique(cluster_group):
            new_centroids.append(X[cluster_group==type].mean(axis=0))
        return np.array(new_centroids)        
    


        



