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

from pvlib import solarposition

# Logging
import logging
logging.basicConfig(level=logging.INFO)

# Import database helper
import db_helper


################################################################################
#                                                                              #
#                        Loading the data from the DB                          #
#                                                                              #
################################################################################


# get the number of accidents
res = db_helper.get_accidents_count()
print(f'Number of Accidents : {res}')
print('\n\n\n')



################################################################################
#                                                                              #
# This section of the script computes additional features.                     #
#                                                                              #
################################################################################

# Find the number of past accidents on each road segment
df = df.sort_values(by=['segmentid', 'datetime']).reset_index(drop=True)
df.loc[:, 'pastaccidents'] = df.groupby('segmentid', as_index=False).accident.cumsum().values

# Convert the road segment class to binary variables
names = dict(zip(df.loc[:, 'class'].unique().tolist(), ['class'+str(x) for x in df.loc[:, 'class'].unique().tolist()]))
binaryclass = pd.get_dummies(df.loc[:, 'class']).rename(columns=names)
df = pd.merge(df, binaryclass, left_index=True, right_index=True)
df = df.drop('class', axis=1)

# Convert the road segment direction to binary variables
names = dict(zip(df.loc[:, 'direction'].unique().tolist(), ['direction'+str(x) for x in df.loc[:, 'direction'].unique().tolist()]))
binarydirection = pd.get_dummies(df.loc[:, 'direction']).rename(columns=names)
df = pd.merge(df, binarydirection, left_index=True, right_index=True)
df = df.drop('direction', axis=1)

# Find the road segment length
df.loc[:, 'roadlength'] = df.geometry.length

# One Hot Econding of month, week, day, weekday, hour

# log
logging.info(f'Additional features added')



################################################################################
#                                                                              #
# This section of the script finds the elevation, zenith and azimuth of the    #
# sun for each example given the location and time of the event.               #
#                                                                              #
################################################################################

# Create a copy of the current data frame
dfcopy = df.loc[:, ['datetime', 'geometry']].copy(deep=True)

# Redefine the geometry of the data frame
dfcopy.loc[:, 'geometry'] = dfcopy.geometry.centroid
dfcopy = dfcopy.set_crs('EPSG:32188')
dfcopy = dfcopy.to_crs('EPSG:4326')

# Find the longitude and latitude of each road segment's centroid
dfcopy.loc[:, 'longitude'] = dfcopy.geometry.x
dfcopy.loc[:, 'latitude'] = dfcopy.geometry.y
dfcopy = dfcopy.drop('geometry', axis=1)

# Find the solar position statistics
dfsolar = solarposition.get_solarposition(dfcopy.datetime, dfcopy.latitude, dfcopy.longitude)

# Find the altitude and azimuth of the sun for each example
df.loc[:, 'elevation'] = dfsolar.apparent_elevation.values
df.loc[:, 'zenith'] = dfsolar.apparent_zenith.values
df.loc[:, 'azimuth'] = dfsolar.azimuth.values

# log
logging.info(f'Solar info done')
