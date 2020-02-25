# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:18:59 2020

@author: jmiller
"""

import pandas as pd
import numpy as np
import os
import networkx as nx
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
    adjusted dataframe
    
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

################
# Top stations #
################
def top_stations(df, n, show = True):
    '''
    Parameters
    ----------
    df: input dataframe
    n: number of top stations to look at
        Finds top starting and ending stations
        Plots top starting and ending stations
    show: whether to print out (default: True)

    Returns
    -------
    top_start_stations, top_end_stations
    
    '''
    station_ids = get_station_ids(df)
    
    top_start_stations = df['start station id'].value_counts()[:n]
    top_end_stations = df['end station id'].value_counts()[:n]
    
    if show:
        print('\n--- Top {} Starting Stations ---'.format(n))
        [print('{}. [{}] {}'.format(
         i, j, station_ids[j]))
         for i, j in enumerate(top_start_stations.index, 1)]
        
        print('\n--- Top {} Ending Stations ---'.format(n))
        [print('{}. [{}] {}'.format(
         i, j, station_ids[j]))
         for i, j in enumerate(top_end_stations.index, 1)]
    
    return top_start_stations, top_end_stations

#####################
# Print top n trips #
#####################
def top_trips(df, n):
    '''
    Parameters
    ----------
    df: adjusted dataframe
    n (int): number of top trips to print
        
    Returns
    -------
    Prints top n trips
    
    '''
    station_ids = get_station_ids(df)
    grouped = df.groupby(['start station id', 'end station id']).count()\
                 ['tripduration']
    grouped = grouped.nlargest(n)
    
    edges = []
    for i in grouped.index:
        edges.append(i)
    for num, (i, j) in enumerate(edges, 1):
        print('{}: [{}] {} to [{}] {}'.format(num,
                                              i, station_ids[i],
                                              j, station_ids[j]))

###############################
# Plots and related functions #
###############################
def plot_legend(df, sta_nums):
    '''
    Parameters
    ----------
    df: adjusted dataframe
    sta_nums: set of station numbers to be plotted
        
    Returns
    -------
    Nothing.
    Adds consistent stylized legend with station IDs/names to graphs.
    
    '''
    station_ids = get_station_ids(df)
    handles = [mpl.patches.Patch(facecolor = 'k',
                                 edgecolor = 'k',
                                 label = '{}: {}'.format(num,
                                                         station_ids[num]))
               for num in sorted(list(sta_nums))]
    leg = plt.legend(handles = handles,
                     loc = 'upper center',
                     bbox_to_anchor = (0.5, -0.05),
                     shadow = True,
                     ncol = 2,
                     handlelength = 0,
                     handletextpad = 0,
                     fancybox = True)
    for item in leg.legendHandles:
        item.set_visible(False)
    leg.get_frame().set_facecolor('lightblue')
    
########################
# Most traveled months #
########################
def most_traveled_months(df, ax):
    '''
    Parameters
    ----------
    df: adjusted dataframe with start_month column added

    Returns
    -------
    bar chart of most traveled months
    '''
    months, rides = np.unique(df['start_month'], return_counts = True)
    
    colors = mpl.cm.summer(rides / max(rides))
    bar = ax.bar(x = [get_month(month) for month in months],
                 height = rides,
                 color = colors,
                 zorder = 2)
    ax.grid(axis = 'y', zorder = 1)
    ax.tick_params(axis = 'x', labelrotation = 70)
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Rides')
    ax.set_title('Total Rides per Month')
    ax.set_facecolor('lightgray')
    return bar

######################
# Most traveled days #
######################
def most_traveled_days(df, ax):
    '''
    Parameters
    ----------
    df: adjusted dataframe with start_days column added

    Returns
    -------
    bar chart of most traveled days
    '''
    days, rides = np.unique(df['start_day'], return_counts = True)
    
    colors = mpl.cm.summer(rides / max(rides))
    bar = ax.bar(x = [get_weekday(day) for day in days],
                 height = rides,
                 color = colors,
                 zorder = 2)
    ax.grid(axis = 'y', zorder = 1)
    ax.tick_params(axis = 'x', labelrotation = 70)
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Total Rides')
    ax.set_title('Total Rides per Day')
    ax.set_facecolor('lightgray')
    return bar

#######################
# Most traveled times #
#######################
def most_traveled_times(df, ax):
    '''
    Parameters
    ----------
    df: adjusted dataframe with start_hour column added

    Returns
    -------
    bar chart of most traveled hours
    '''
    hours, rides = np.unique(df['start_hour'], return_counts = True)
    
    colors = mpl.cm.summer(rides / max(rides))
    bar = ax.bar(x = hours,
                 height = rides,
                 color = colors,
                 zorder = 2)
    ax.grid(axis = 'y', zorder = 1)
    ax.set_xlabel('Hour of the Day')
    ax.set_ylabel('Total Rides')
    ax.set_xticks(np.arange(0, 23, 3))
    ax.set_title('Total Rides per Hour')
    ax.set_facecolor('lightgray')
    return bar

###################
# Rides by gender #
###################
def rides_by_gender(df, ax):
    '''
    Parameters
    ----------
    df: adjusted dataframe

    Returns
    -------
    bar chart of rides by gender
    '''
    gender, rides = np.unique(df['gender'], return_counts = True)
    
    colors = mpl.cm.summer(rides / max(rides))
    bar = ax.bar(x = gender,
                 height = rides,
                 color = colors,
                 zorder = 2)
    ax.grid(axis = 'y', zorder = 1)
    ax.tick_params(axis = 'x', labelrotation = 0)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Male', 'Female', 'Prefer not to say'])
    ax.set_xlabel('Gender')
    ax.set_ylabel('Total Rides')
    ax.set_title('Total Rides per Gender')
    ax.set_facecolor('lightgray')
    return bar

################
# Matrix plots #
################
def matrix_plot():
    '''
    Parameters
    ----------
    df: adjusted dataframe

    Returns
    -------
    2 x 2 subplots
    '''
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2,
                                                 ncols = 2,
                                                 figsize = (15, 15))
    most_traveled_months(df, ax1)
    most_traveled_days(df, ax2)
    most_traveled_times(df, ax3)
    rides_by_gender(df, ax4)
    plt.tight_layout()

##########################
# Directed network graph #
##########################
def network_graph(df, n):
    '''
    Parameters
    ----------
    df: adjusted dataframe
    n: (int) number of rides to plot

    Returns
    -------
    network graph of top n routes
    '''
    grouped = df.groupby(['start station id', 'end station id']).count()\
                     ['tripduration']
    grouped = grouped.nlargest(n)
    
    station_numbers = set()
    edge_collection = []
    weights = []
    for i in grouped.index:
        edge_collection.append(i)
        weights.append(grouped[i])
        station_numbers.update(i)
    
    G = nx.DiGraph()
    G.add_nodes_from(station_numbers)
    
    pos = nx.layout.circular_layout(G)
    M = G.number_of_edges()
    edge_colors = range(2, M + 2)
    edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
    
    plt.figure(figsize = (15, 15))
    nodes = nx.draw_networkx_nodes(G,
                                   pos,
                                   node_size = 5,
                                   node_color = 'black')
    edges = nx.draw_networkx_edges(G,
                                   pos,
                                   node_size = 5,
                                   arrowstyle = '-|>',
                                   arrowsize = 20,
                                   edgelist = edge_collection,
                                   edge_color = weights,
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
    plot_legend(df, station_numbers)
    plt.axis('equal')
    plt.title('Top {} Blue Bike Routes'.format(n),
              fontdict = {'fontsize': 25,
                          'fontweight': 'bold'},
              pad = -25)
    plt.show()

#######################
# Plot rides per dock #
#######################
def rides_per_dock(df, sta_df, n):
    '''
    Parameters
    ----------
    df: adjusted dataframe
    station_df: dataframe with station names and total docks
    n: (int) number of stations to look at
    start: (bool) whether to look at starting (default) or ending stations

    Returns
    -------
    bar chart of rides per station for top n -ending- stations
    '''
    station_ids = get_station_ids(df)
    
    # Concatenate top starting/ending stations on station ID
    top_df = pd.concat(top_stations(df, -1, False), axis = 1).reset_index()
    top_df = top_df.nlargest(n, 'end station id')
    top_df.columns = ['ID', 'Start rides', 'End rides']
    
    # Station_df uses names instead of numbers, so need to join on names
    top_df['Name'] = top_df['ID'].apply(lambda x: station_ids[x])
    top_df = top_df.join(sta_df.set_index('Name'), on = 'Name')
    
    # Some stations have 0 docks. Delete, but worth investigating separately
    top_df = top_df[top_df['Total docks'] != 0]
    
    # Calculate rides per dock for starting/ending stations
    top_df['Starting rides per dock'] = \
        top_df.apply(lambda x: x['Start rides']/x['Total docks'], axis = 1)
    top_df['Ending rides per dock'] = \
        top_df.apply(lambda x: x['End rides']/x['Total docks'], axis = 1)
    top_df = top_df.sort_values(by = 'Ending rides per dock')
    
    # Pull out useful columns
    top_df = top_df[['ID', 'Ending rides per dock',
                     'Starting rides per dock']].set_index('ID')
    
    # Plot
    top_df.plot(figsize = (15, 15),
                kind = 'barh',
                colormap = 'coolwarm',
                zorder = 2)
    plt.title('Rides per Dock',
              fontdict = {'fontsize': 25,
                          'fontweight': 'bold'},
              pad = -25)
    plt.xlabel('Rides', fontsize = 15)
    plt.ylabel('Station Number', fontsize = 15)
    plt.grid(which = 'major', axis = 'x')
    plt.tick_params(axis = 'both', labelsize = '15')

    # Shade every other station
    for i, j in enumerate(top_df.index):
        if i % 2 == 0:
            plt.axhspan(i - 0.5,
                        i + 0.5,
                        facecolor = 'lightgray',
                        zorder = 1)
    
    # Plot 2 legends
    ax = plt.gca()
    ax.add_artist(plt.legend(loc = 'lower right'))
    station_numbers = set()
    for i in top_df.index:
        station_numbers.add(i)
    plot_legend(df, station_numbers)
    plt.show()

#################
# DATA ANALYSIS #
#################
##############################
# Load data, setup dataframe #
##############################
df, df_copy = load_data()
df = setup_dataframe(df)
df = remove_outliers(df)

################
# Explore Data #
################
print('\n--- Preview of Data Set ---')
print(df.head())
print('\n--- Example Entry ---')
print(df.iloc[0])
print('\n--- Descriptive Statistics ---')
print(df.describe().T)

################
# Top stations #
################
top_start_stations, top_end_stations = top_stations(df, 5, True)

#################
# Bike stations #
#################
station_df = pd.read_csv('Data/current_bluebikes_stations.csv', skiprows = 1)
print()
print(station_df.head())

#########
# Plots #
#########
matrix_plot()
network_graph(df, 100)
rides_per_dock(df, station_df, 10)