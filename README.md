# GENETIC ALGORITHM FOR MULTI-DEPOT DRONE INSPECTION PROBLEM
Uses K-means algorithm to choose charge piles, and Cohesion clustering to find towers for every charge pile.
Genetic algorithm is performed separately for each cluster.
We are currently working on global optimization for ambiguous points that are far away from its cluster centroids. A possible solution is to find out how much these points value to adjacent clusters by conducting fast experiments on w or w/o these points for each cluster.
