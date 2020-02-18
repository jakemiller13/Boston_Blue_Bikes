# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:18:59 2020

@author: jmiller
"""

# TODO scatterplot matrix not looking great
# TODO use start time + trip duration to determine when bikes are dropped off
# TODO find top rental days
# TODO deep dive into most active station - do bikes depart more than arrive?
# TODO when do they arrive/leave

import pandas as pd
import numpy as np
import os
import networkx as nx
import plotly.express as px
from plotly.offline import plot
import matplotlib as mpl
import matplotlib.pyplot as plt

# Set seed
np.random.seed(42)

#####################
# Utility Functions #
#####################
def load_data():
    '''
    Parameters
    ----------
    None
        Loads data into dataframe and returns df and df_copy

    Returns
    -------
    df, copy of df

    '''
    df = pd.DataFrame()
    
    for file in os.listdir('Data'):
        if '2019' in file:
            print('...Reading file "{}"...'.format(file))
            df = df.append(pd.read_csv('Data/' + file,
                                       parse_dates = ['starttime',
                                                      'stoptime']))
    df = df.reset_index(drop = True)
    
    return df, df.copy()

def get_weekday(day_num):
    '''
    Parameters
    ----------
    day_num: int
        Returns string of day of week

    Returns
    -------
    Day of week (string)

    '''
    day = {0: 'Monday',
           1: 'Tuesday',
           2: 'Wednesday',
           3: 'Thursday',
           4: 'Friday',
           5: 'Saturday',
           6: 'Sunday'}
    
    return day[int(day_num)]

def get_month(month_num):
    '''
    Parameters
    ----------
    month_num: int
        Returns string of month

    Returns
    -------
    Month (string)

    '''
    month = {1: 'January',
             2: 'February',
             3: 'March',
             4: 'April',
             5: 'May',
             6: 'June',
             7: 'July',
             8: 'August',
             9: 'September',
             10: 'October',
             11: 'November',
             12: 'December'}
    
    return month[int(month_num)]

def get_station_ids(df):
    '''
    Parameters
    ----------
    df: dataframe to gather unique stations ids

    Returns
    -------
    dict: {station ID: station name}

    '''
    id_df = df[['start station id', 'start station name']].\
                drop_duplicates().reset_index(drop = True)
    id_dict = {id_df['start station id'].loc[i]: 
               id_df['start station name'].loc[i]
               for i in range(id_df.shape[0])}
    
    return id_dict
    
#######################
# Setup other columns #
#######################
def setup_dataframe(df):
    '''
    Parameters
    ----------
    df: input dataframe.
        Adds new columns for start_hour, start_day

    Returns
    -------
    Adjusted dataframe
    
    '''
    # Ride start hour
    print('\n...Collecting starting hour...')
    df['start_hour'] = df['starttime'].apply(lambda x: x.hour)
    
    # Ride start day
    print('\n...Collecting start day...')
    df['start_day'] = df['starttime'].apply(lambda x: x.weekday())
    
    # Ride start month
    print('\n...Collecting starting month...')
    df['start_month'] = df['starttime'].apply(lambda x: x.month)
    
    return df

####################################
# Quantitative features to look at #
####################################
features = ['tripduration',
            'start_hour',
            'birth year']

def remove_outliers(df):
    '''
    Parameters
    ----------
    df: input dataframe.
        Removes trips >24 hours
        Removes rides with birth year <1920
        Respect to anyone 100+ still riding but...

    Returns
    -------
    Trimmed dataframe
    
    '''
    long_trips = df[df['tripduration'] > 24 * 60 * 60]
    too_old = df[df['birth year'] < 1920]
    df = df.drop(index = long_trips.index.append(too_old.index))
    return df

# ###################
# # Remove outliers #
# ###################
# plt.figure()
# plt.hist(df['tripduration'])
# plt.title('Trip Duration (before removing outliers')
# plt.xlabel('Duration (seconds)')
# plt.ylabel('Count')
# plt.show()

# # Crazy numbers. Remove anything over a day
# std = np.std(df['tripduration'])
# mean = np.mean(df['tripduration'])

# outliers = df[df['tripduration'] > 86400]

# print('\n--- Number of outliers removed: {} ---'.format(outliers.shape[0]))
# df = df.drop(outliers.index).reset_index(drop = True)

# plt.figure()
# plt.hist(df['tripduration'], bins = 23)
# plt.title('Trip Duration (after removing 1st outliers)')
# plt.xlabel('Duration (seconds)')
# plt.ylabel('Count')
# plt.show()

# # Still crazy. Only look at hours with at least 1000 data points per hour
# counts, bin_edges = np.histogram(df['tripduration']/3600,
#                                  bins = np.arange(0, 24))
# print('\nHours with >1000 data points: {}'.format(np.where(counts > 1000)[0]))

# outliers = df[df['tripduration'] > 3 * 60 * 60]

# print('--- Number of outliers removed: {} ---'.format(outliers.shape[0]))
# df = df.drop(outliers.index).reset_index(drop = True)

# plt.figure()
# plt.hist(df['tripduration'], bins = 25)
# plt.title('Trip Duration (after removing outliers)')
# plt.xlabel('Duration (seconds)')
# plt.ylabel('Count')
# plt.show()

# ##################
# # Scale features #
# ##################
# scaler = MinMaxScaler()
# scaled_df = pd.DataFrame(data = scaler.fit_transform(df[features]),
#                          columns = features)

# #############
# # 5% sample #
# #############
# samples = np.random.choice(scaled_df.shape[0], scaled_df.shape[0]//100)
# sample_df = scaled_df.iloc[samples]

# #############################
# # k-means run on sample set #
# #############################

# # Setup metrics
# sse = []
# scores = []
# cluster_range = np.arange(2, 10)

# # Train multiple k-means and test silhouette coefficients, elbow
# for i in cluster_range:
#     print('\n...Fitting {} clusters...'.format(i))
#     kmeans = KMeans(n_clusters = i,
#                     init = 'k-means++',
#                     n_init = 10,
#                     max_iter = 300,
#                     tol = 0.0001,
#                     verbose = 0)
#     kmeans.fit(sample_df[features])
    
#     score = metrics.silhouette_score(sample_df[features],
#                                       kmeans.labels_,
#                                       metric = 'euclidean',
#                                       sample_size = len(sample_df[features]))
#     scores.append(score)
#     sse.append(kmeans.inertia_)
#     print('Number of clusters:', i)
#     print('Silhouette score:', score)
#     print('Inertia:', kmeans.inertia_)

# plt.figure()
# plt.plot(cluster_range, sse, '-o')
# plt.xlabel('Number of clusters')
# plt.ylabel('Sum of squared distance')
# plt.show()

# #####################
# # Best: 4 centroids #
# #####################
# best_k = KMeans(n_clusters = 4,
#                 init = 'k-means++',
#                 n_init = 10,
#                 max_iter = 300,
#                 tol = 0.0001,
#                 verbose = 0)
# best_k.fit(df[features])

# centers = best_k.cluster_centers_
# clusters, cluster_counts = np.unique(best_k.labels_, return_counts = True)

# print('\n--- Cluster centers ---\n{}'.format(
#       pd.DataFrame(data = centers, columns = features)))
# print('\n--- Cluster counts ---\n{}'.format(
#       pd.DataFrame(data = cluster_counts,
#                    columns = ['Counts'],
#                    index = clusters)))

################
# Top stations #
################
def top_stations(df, n):
    '''
    Parameters
    ----------
    df: input dataframe
    n: number of top stations to look at
        Finds top starting and ending stations
        Plots top starting and ending stations

    Returns
    -------
    N/A
    
    '''
    station_ids = get_station_ids(df)
    
    top_start_stations = df['start station id'].value_counts()[:n]
    top_end_stations = df['end station id'].value_counts()[:n]
    
    print('\n--- Top {} Starting Stations ---'.format(n))
    [print('{}. {}'.format(
     i, station_ids[j])) for i, j in enumerate(top_start_stations.index, 1)]
    
    print('\n--- Top {} Ending Stations ---'.format(n))
    [print('{}. {}'.format(
     i, station_ids[j])) for i, j in enumerate(top_end_stations.index, 1)]

######################
# Most traveled days #
######################
def most_traveled_months(df):
    '''
    Parameters
    ----------
    df: adjusted dataframe with start_month column added

    Returns
    -------
    bar chart of most traveled months
    '''
    months, rides = np.unique(df['start_month'], return_counts = True)
    fig = px.bar(x = [get_month(month) for month in months],
                 y = rides,
                 color = rides)
    fig.update_layout(title = 'Total Rides per Month',
                      xaxis_title = 'Month',
                      yaxis_title = 'Total Rides')
    plot(fig)

######################
# Most traveled days #
######################
def most_traveled_days(df):
    '''
    Parameters
    ----------
    df: adjusted dataframe with start_days column added

    Returns
    -------
    bar chart of most traveled days
    '''
    days, rides = np.unique(df['start_day'], return_counts = True)
    fig = px.bar(x = [get_weekday(day) for day in days],
                 y = rides,
                 color = rides)
    fig.update_layout(title = 'Total Rides per Day',
                      xaxis_title = 'Weekday',
                      yaxis_title = 'Total Rides')
    plot(fig)

#######################
# Most traveled times #
#######################
def most_traveled_times(df):
    '''
    Parameters
    ----------
    df: adjusted dataframe with start_hour column added

    Returns
    -------
    bar chart of most traveled hours
    '''
    hours, rides = np.unique(df['start_hour'], return_counts = True)
    fig = px.bar(x = hours,
                 y = rides,
                 color = rides)
    fig.update_layout(title = 'Total Rides per Hour',
                      xaxis_title = 'Hour of the Day',
                      yaxis_title = 'Total Rides')
    plot(fig)

###################
# Rides by gender #
###################
def rides_by_gender(df):
    '''
    Parameters
    ----------
    df: adjusted dataframe

    Returns
    -------
    bar chart of rides by gender
    '''
    gender, rides = np.unique(df['gender'], return_counts = True)
    fig = px.bar(x = gender,
                 y = rides,
                 color = rides)
    fig.update_layout(title = 'Total Rides per Gender',
                      xaxis_title = 'Gender',
                      yaxis_title = 'Total Rides')
    plot(fig)
    
#################
# DATA ANALYSIS #
#################
##############################
# Load data, setup dataframe #
##############################
df, df_copy = load_data()
df = setup_dataframe(df)
remove_outliers(df)

##########################
# Descriptive statistics #
##########################
print('\n--- Descriptive Statistics ---')
print(df[features].describe().to_string())

######################
# Scatterplot Matrix #
######################
# sns.pairplot(sample_df[features])

top_stations(df, 5)
most_traveled_months(df)
most_traveled_days(df)
most_traveled_times(df)
rides_by_gender(df)

##################
# DIRECTED GRAPH #
##################
top_start_stations = df['start station id'].value_counts()[:25]
top_end_stations = df['end station id'].value_counts()[:25]
top_stations = set(top_start_stations.index.append(top_end_stations.index))

grouped = df.groupby(['start station id', 'end station id']).count()\
                     ['tripduration']
grouped = grouped.nlargest(100)
trip_direction = grouped.index

station_numbers = set()
edges = []
for i in grouped.index:
    edges.append(i)
    station_numbers.update(i)

station_names = get_station_ids(df)
labels = ['{}: {}'.format(num, station_names[num])
          for num in sorted(list(station_numbers))]

G = nx.DiGraph()
G.add_nodes_from(station_numbers)
G.add_edges_from(edges)

pos = nx.layout.circular_layout(G)
M = G.number_of_edges()
edge_colors = range(2, M + 2)
edge_alphas = [(5 + i) / (M + 4) for i in range(M)]

fig = plt.figure(figsize = (15, 15))
nodes = nx.draw_networkx_nodes(G,
                               pos,
                               node_size = 5,
                               node_color = 'black')
edges = nx.draw_networkx_edges(G,
                               pos,
                               node_size = 5,
                               arrowstyle = '-|>',
                               arrowsize = 20,
                               edge_color = edge_colors,
                               edge_cmap = plt.cm.rainbow,
                               width = 2)

for i in range(M):
    edges[i].set_alpha(edge_alphas[i])
pc = mpl.collections.PatchCollection(edges,
                                     cmap = plt.cm.rainbow)
pc.set_array(edge_colors)
cbar = plt.colorbar(pc,
                    fraction = 0.04,
                    pad = 0.01)
cbar.ax.tick_params(labelsize = 20) 
nx.draw_networkx_labels(G,
                        pos,
                        font_size = 20)

ax = plt.gca()
ax.set_axis_off()
leg = ax.legend(labels = labels,
                loc='upper center',
                bbox_to_anchor = (0.5, -0.05),
                shadow = True,
                ncol = 2,
                handlelength = 0,
                handletextpad = 0,
                fancybox = True)
for item in leg.legendHandles:
    item.set_visible(False)
plt.axis('equal')
plt.title('Top 100 Blue Bike Routes',
          fontdict = {'fontsize': 25})
plt.show()