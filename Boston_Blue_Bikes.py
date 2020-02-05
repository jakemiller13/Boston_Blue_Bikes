# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:18:59 2020

@author: jmiller
"""

# TODO histogram of starting hours
# TODO histogram of trip duration
# TODO limit to +2 stds above mean

import pandas as pd
import numpy as np
import os
from sklearn import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#############
# Load data #
#############
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

# Capture hour of ride start
df['start_hour'] = df['starttime'].apply(lambda x: pd.to_datetime(x).hour)

#######################
# Features to look at #
#######################
features = ['tripduration',
            'start_hour',
            'start station latitude',
            'start station longitude']

##########################
# Descriptive statistics #
##########################
print('\n--- Descriptive Statistics ---')
print(df[features].describe().to_string())

###################
# Remove outliers #
###################
plt.figure()
plt.hist(df[features]['tripduration'])
plt.xlabel('Trip Duration')
plt.ylabel('Count')
plt.show()

std = np.std(df['tripduration'])
mean = np.mean(df['tripduration'])

outliers = df.shape[0] - df[df['tripduration'] < mean + 2 * std]\
                           [features].shape[0]
print('\n--- Number of outliers removed: {} ---'.format(outliers))
df = df[df['tripduration'] < mean + 2 * std]

#############
# 5% sample #
#############
samples = np.random.choice(df.shape[0], df.shape[0]//20)
sample_df = df.iloc[samples]

#############################
# k-means run on sample set #
#############################

# Setup metrics
sse = []
scores = []
cluster_range = np.arange(2, 10)

# Train multiple k-means and test silhouette coefficients, elbow
for i in cluster_range:
    print('\n...Fitting {} clusters...'.format(i))
    kmeans = KMeans(n_clusters = i,
                    init = 'k-means++',
                    n_init = 10,
                    max_iter = 300,
                    tol = 0.0001,
                    verbose = 0)
    kmeans.fit(sample_df[features])
    
    score = metrics.silhouette_score(sample_df[features],
                                      kmeans.labels_,
                                      metric = 'euclidean',
                                      sample_size = len(sample_df[features]))
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

#####################
# Best: 4 centroids #
#####################
best_k = KMeans(n_clusters = 4,
                init = 'k-means++',
                n_init = 10,
                max_iter = 300,
                tol = 0.0001,
                verbose = 1)
best_k.fit(df[features])

centers = best_k.cluster_centers_
clusters, cluster_counts = np.unique(best_k.labels_, return_counts = True)

print('--- Cluster centers ---\n{}'.format(
      pd.DataFrame(data = centers, columns = features)))
print('\n--- Cluster counts ---\n{}'.format(
      pd.DataFrame(data = cluster_counts,
                   columns = ['Counts'],
                   index = clusters)))

##################
# Starting hours #
##################
plt.figure()
n, bins, patches = plt.hist(df['start_hour'], bins = 23)
patches[8].set_fc('r')
patches[17].set_fc('r')
plt.grid()
plt.xlabel('Starting hour of ride')
plt.ylabel('Counts')
plt.show()

################
# Top stations #
################
# Most used starting/endings stations
start_station, start_station_counts = np.unique(df['start station id'], 
                                                return_counts = True)
end_station, end_station_counts = np.unique(df['end station id'], 
                                            return_counts = True)
top_start_station = start_station[np.argmax(start_station_counts)]
top_end_station = end_station[np.argmax(end_station_counts)]

print('--- Most Used Start Station ---\n{}'.format(
      df[df['start station id'] == top_start_station]
           ['start station name'].iloc[0]))

print('\n--- Most Used End Station ---\n{}'.format(
      df[df['end station id'] == top_end_station]
           ['end station name'].iloc[0]))