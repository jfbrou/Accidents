################################################################################
#                                                                              #
# This section of the script imports creates additional features to our dataset#
#                                                                              #
################################################################################

# Import libraries
import os
import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.wkt import loads as wkt_loads

from pvlib import solarposition

# Logging
import logging
logging.basicConfig(level=logging.INFO)

# Import database helper
import db_helper

# Find the current working directory
path = os.getcwd()

# Create a folder that contains all data files
if os.path.isdir(os.path.join(path, 'Data')) == False:
    raise Exception('Data directory does not exist, run retrieve script')
data_dir_path = os.path.join(path, 'Data')


################################################################################
#                                                                              #
#                        Loading the data from the DB                          #
#                                                                              #
################################################################################


# get the processed accidents
df = db_helper.get_accidents(LIMIT=10000)
df['accident_geometry'] = df['accident_geometry'].apply(wkt_loads)
df['road_segment_geometry'] = df['road_segment_geometry'].apply(wkt_loads)
df = gpd.GeoDataFrame(df)


################################################################################
#                                                                              #
# This section of the script computes additional features.                     #
#                                                                              #
################################################################################


df = df.set_geometry('road_segment_geometry')

# Compute the road segment length
df.loc[:, 'road_segment_length'] = df['road_segment_geometry'].length

# Compute the area of the road segment's convex hull
df.loc[:, 'road_segment_convexhull'] = df['road_segment_geometry'].convex_hull.area

df = df.set_geometry('accident_geometry')


# Convert the road_segment_class to binary variables
names = dict(zip(df.loc[:, 'road_segment_class'].unique().tolist(), ['road_segment_class'+str(x) for x in df.loc[:, 'road_segment_class'].unique().tolist()]))
binary = pd.get_dummies(df.loc[:, 'road_segment_class']).rename(columns=names)
df = pd.merge(df, binary, left_index=True, right_index=True)
df = df.drop('road_segment_class', axis=1)

# Convert the road segment road_segment_direction to binary variables
names = dict(zip(df.loc[:, 'road_segment_direction'].unique().tolist(), ['road_segment_direction'+str(x) for x in df.loc[:, 'road_segment_direction'].unique().tolist()]))
binary = pd.get_dummies(df.loc[:, 'road_segment_direction']).rename(columns=names)
df = pd.merge(df, binary, left_index=True, right_index=True)
df = df.drop('road_segment_direction', axis=1)

# Convert months to binary variables
names = dict(zip(df['accident_datetime'].dt.month.unique().tolist(), ['month'+str(x) for x in df['accident_datetime'].dt.month.unique().tolist()]))
binary = pd.get_dummies(df['accident_datetime'].dt.month).rename(columns=names)
df = pd.merge(df, binary, left_index=True, right_index=True)

# Convert weeks to binary variables
names = dict(zip(list(df['accident_datetime'].dt.isocalendar().week.unique()), ['week'+str(x) for x in list(df['accident_datetime'].dt.isocalendar().week.unique())]))
binary = pd.get_dummies(df['accident_datetime'].dt.isocalendar().week).rename(columns=names)
df = pd.merge(df, binary, left_index=True, right_index=True)

# Convert days to binary variables
names = dict(zip(df['accident_datetime'].dt.day.unique().tolist(), ['day'+str(x) for x in df['accident_datetime'].dt.day.unique().tolist()]))
binary = pd.get_dummies(df['accident_datetime'].dt.day).rename(columns=names)
df = pd.merge(df, binary, left_index=True, right_index=True)

# Convert weekdays to binary variables
names = dict(zip(df['accident_datetime'].dt.weekday.unique().tolist(), ['weekday'+str(x) for x in df['accident_datetime'].dt.weekday.unique().tolist()]))
binary = pd.get_dummies(df['accident_datetime'].dt.weekday).rename(columns=names)
df = pd.merge(df, binary, left_index=True, right_index=True)

# Convert hours to binary variables
names = dict(zip(df['accident_datetime'].dt.hour.unique().tolist(), ['hour'+str(x) for x in df['accident_datetime'].dt.hour.unique().tolist()]))
binary = pd.get_dummies(df['accident_datetime'].dt.hour).rename(columns=names)
df = pd.merge(df, binary, left_index=True, right_index=True)


################################################################################
#                                                                              #
# This section of the script finds the elevation, zenith and azimuth of the    #
# sun for each example given the location and time of the event.               #
#                                                                              #
################################################################################

# Create a copy of the current data frame
dfcopy = df.loc[:, ['accident_datetime', 'accident_geometry']].copy(deep=True)

# Find the longitude and latitude of each road segment's centroid
dfcopy.loc[:, 'longitude'] = dfcopy['accident_geometry'].x
dfcopy.loc[:, 'latitude'] = dfcopy['accident_geometry'].y
dfcopy = dfcopy.drop('accident_geometry', axis=1)

# Find the solar position statistics
dfsolar = solarposition.get_solarposition(dfcopy['accident_datetime'], dfcopy.latitude, dfcopy.longitude)

# Find the altitude and azimuth of the sun for each example
df.loc[:, 'elevation'] = dfsolar.apparent_elevation.values
df.loc[:, 'zenith'] = dfsolar.apparent_zenith.values
df.loc[:, 'azimuth'] = dfsolar.azimuth.values



################################################################################
#                                                                              #
# This section of the script randomly samples the negative examples.           #
#                                                                              #
################################################################################

# Create a data frame of randomly sampled negative examples
negative = df.sample(n=df.shape[0], replace=True, random_state=0)

# Randomly alter the time of negative examples
np.random.seed(0)
negative.loc[:, 'randomhours'] = np.random.normal(scale=df.accident_datetime.dt.hour.std(), size=negative.shape[0]).astype('int64')
negative.loc[:, 'accident_datetime'] = negative.accident_datetime+negative.randomhours.transform(lambda x: datetime.timedelta(hours=x))
negative = negative.drop('randomhours', axis=1)

# Randomly alter the date of negative examples
np.random.seed(0)
negative.loc[:, 'randomdays'] = np.random.normal(scale=df.accident_datetime.transform(lambda x: x.timetuple().tm_yday).std(), size=negative.shape[0]).astype('int64')
negative.loc[:, 'accident_datetime'] = negative.accident_datetime+negative.randomdays.transform(lambda x: datetime.timedelta(days=x))
negative = negative.drop('randomdays', axis=1)

# Drop the dates that preceed the year 2012 or exceed the year 2019 in UTC time
negative = negative.loc[(negative.accident_datetime > pd.Timestamp(2012, 1, 1, 4)) & (negative.accident_datetime < pd.Timestamp(2020, 1, 1, 5)), :]

# Drop duplicated observations
negative = negative.drop_duplicates()

# Create the df and negative labels
df.loc[:, 'accident'] = 1
negative.loc[:, 'accident'] = 0

# Append the positive and negative example data frames
df = df.append(negative, ignore_index=True)

# Drop duplicated negative examples
df.loc[:, 'duplicate'] = df.duplicated(keep=False)
df = df.loc[(df.duplicate == False) | (df.accident == 1), :].drop('duplicate', axis=1)

# log
print("Length of set : ", len(df))
print("Number of positives : ", len(df.loc[df['accident'] == 1]))
print("Number of negatives : ", len(df.loc[df['accident'] == 0]))

################################################################################
#                                                                              #
# This section of the script saves the data frame.                             #
#                                                                              #
################################################################################

# remove geometry
df = df.loc[:, df.columns[~df.columns.isin(['accident_geometry', 'road_segment_geometry'])]]

# Save the data
df.to_csv(os.path.join(data_dir_path, 'data.csv'), index=False)
