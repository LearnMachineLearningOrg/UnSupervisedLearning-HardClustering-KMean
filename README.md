# UnSupervisedLearning-HardClustering-KMean
This repository consists of implementation of KMean hard clustering machine learning algorithm

Machine Learning concept being worked on:

1. Clustering is a machine learning algorithm that falls under Unsupervised learning
2. As this is Unsuperised learning, the dataset that we will be working on will only have features and no label. 
3. In clustering, we will be preparing clusters, based on the similarity of the features among the datapoints / examples
4. In clustering technique we have two types of clustering 
        a. Soft clustering: In this type of culstering there will be some datapoints that belongs to multiple cuslters with various degrees of probability
        b. Hard clustering: In this type of clustering every datapoint belongs to only one cluster. There is no overlap
5. Examples of the problems that clustering machine learning algorithm attempts to targets are 
        a. Market segmentation
        b. Recommendation engines
        c. Market segmentation
        d. Social network analysis
        e. Image segmentation

In this exmple we will cluster customers based on their annual income and spending score.

1. Dataset being used: mall.csv
2. Features being used: Annual income, Spending score
3. Use Elbow method with Euclidean distance measure to predict the number of clusters
4. Once the number of clusters are identified then execute the KMeans algorithm to cluster customers based on their annual income and spending score

Python modules being used:

1. os: This module provides a portable way of using operating system dependent functionality.
2. pandas: Pandas provide high-performance data manipulation and analysis tool using its powerful data structures.
3. numpy: NumPy is the fundamental package for scientific computing with Python.
4. matplotlib: matplotlib is a plotting library for the Python programming language.
5. sklearn: It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN
