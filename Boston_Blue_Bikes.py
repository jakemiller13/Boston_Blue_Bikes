# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:18:59 2020

@author: jmiller
"""

# TODO take random sample of entire population
# TODO figure out optimal cluster number
# TODO rerun over entire population

import pandas as pd
import numpy as np
import os
from sklearn import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load data
def load_data():
    '''
    Loads and returns data. Only run if necessary
    '''
    df = pd.DataFrame()
    
    for file in os.listdir('Data'):
        print('...Reading file "{}"...'.format(file))
        df = df.append(pd.read_csv('Data/' + file,
                                   parse_dates = ['starttime', 'stoptime']))
    
    df = df.reset_index(drop = True)
    return df

try:
    if df.empty:
        df = load_data()
    else:
        pass
except NameError:
    df = load_data()
except ValueError:
    pass

# Add columns to capture categorical time
df['start_hour'] = np.nan

# Capture hour of ride start
df['start_hour'] = df['starttime'].apply(lambda x: pd.to_datetime(x).hour)

# Only look at January to see if KMeans runs to completion
df = df[df['starttime'] < np.datetime64('2019-01-10')]


###########
# k-means #
###########

# Columns to use in clustering - all categorical
features = ['tripduration',
            'start_hour',
            'start station latitude',
            'start station longitude']

# One-hot encode categorical variables
# df = pd.get_dummies(df, columns = categories)
# features = []
# for i in categories:
#     features.extend([j for j in df.columns if i in j])

# Setup metrics
sse = []
scores = []
cluster_range = np.arange(2, 10)

# Train multiple k-means and test silhouette coefficients, elbow
for i in cluster_range:
    print('\n...Fitting {} clusters...'.format(i))
    kmeans = KMeans(init = 'k-means++',
                    n_clusters = i,
                    n_init = 10,
                    verbose = 0)
    kmeans.fit(df[features])
    
    score = metrics.silhouette_score(df[features],
                                      kmeans.labels_,
                                      metric = 'euclidean',
                                      sample_size = len(df[features]))
    scores.append(score)
    sse.append(kmeans.inertia_)
    print('Number of clusters:', i)
    print('Silhouette score:', score)
    print('Inertia:', kmeans.inertia_)

plt.figure()
plt.plot(cluster_range, sse, '-o')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared distance')
plt.show()

# 5 centroids appears best
best_k = KMeans(init = 'k-means++',
                n_clusters = 5,
                n_init = 10)
best_k.fit(df)

best_k.cluster_centers_
best_k.labels_