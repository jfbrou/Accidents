################################################################################
#                                                                              #
# This section of the script imports libraries, creates directories and        #
# retrieves the road traffic accidents, road segments and weather data.        #
#                                                                              #
################################################################################

# Import libraries
import os
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from pvlib import solarposition

# Logging
import logging
logging.basicConfig(level=logging.INFO)

# Import database helper
import db_helper

# Find the current working directory
path = os.getcwd()

# Create a folder that contains all figures
if os.path.isdir(os.path.join(path, 'Figures')) == False:
    os.mkdir('Figures')
figures_dir_path = os.path.join(path, 'Figures')


def draw_as_pdf(geometries, out_path):
    """
        Function to draw geometries on a pdf
    """

    if os.path.exists(out_path) == False:

        # plot
        fig, ax = plt.subplots()
        geometries.plot(ax=ax, color='teal', markersize=0.1)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_path, format='pdf')
        plt.close()

        # log
        logging.info(f'File created {out_path}')

    else:
        logging.info(f'Already exists {out_path}')



def get_weighted_weather(weather_stations_data):

    # convert to dict
    weather_records = weather_stations_data.to_dict(orient='records')

    print(weather_records)




################################################################################
#                                                                              #
#                        Loading the data from the DB                          #
#                                                                              #
################################################################################


# get the number of accidents
res = db_helper.get_accidents_count()
print(f'Number of Accidents : {res}')
print('\n\n\n')

# get accidents matched with road segments
accidents_roadsegments = db_helper.match_accidents_with_road_segments(
    NBR_ACCIDENTS_IN_PROCESSED_BATCH=2,
    OFFSET=0
)
print(accidents_roadsegments)
print('\n\n\n')

# get accidents matched with weather data
accidents_weatherrecords = db_helper.match_accidents_with_weather_records(
    NBR_ACCIDENTS_IN_PROCESSED_BATCH=2,
    OFFSET=0
)
print(accidents_weatherrecords)
print('\n\n\n')


raise Exception('stop')



################################################################################
#                                                                              #
#                                 Produce PDFs                                 #
#                                                                              #
################################################################################

# write
out_path = os.path.join(figures_dir_path, 'accidents_2019.pdf')
draw_as_pdf(accidents, out_path)

# write
out_path = os.path.join(figures_dir_path, 'segments.pdf')
draw_as_pdf(road_segments, out_path)


################################################################################
#                                                                              #
# This section of the script randomly samples the negative examples.           #
#                                                                              #
################################################################################

# Create a data frame of randomly sampled negative examples
negative = positive.sample(n=4*positive.shape[0], replace=True, random_state=0)

# Randomly alter the time of negative examples
np.random.seed(0)
negative.loc[:, 'randomhours'] = np.random.normal(scale=positive.datetime.dt.hour.std(), size=negative.shape[0]).astype('int64')
negative.loc[:, 'datetime'] = negative.datetime+negative.randomhours.transform(lambda x: datetime.timedelta(hours=x))
negative = negative.drop('randomhours', axis=1)

# Randomly alter the date of negative examples
np.random.seed(0)
negative.loc[:, 'randomdays'] = np.random.normal(scale=positive.datetime.transform(lambda x: x.timetuple().tm_yday).std(), size=negative.shape[0]).astype('int64')
negative.loc[:, 'datetime'] = negative.datetime+negative.randomdays.transform(lambda x: datetime.timedelta(days=x))
negative = negative.drop('randomdays', axis=1)

# Drop the dates that preceed the year 2012 or exceed the year 2019 in UTC time
negative = negative.loc[(negative.datetime > pd.Timestamp(2012, 1, 1, 4)) & (negative.datetime < pd.Timestamp(2020, 1, 1, 5)), :]

# Drop duplicated observations
negative = negative.drop_duplicates()

# Create the positive and negative labels
positive.loc[:, 'accident'] = 1
negative.loc[:, 'accident'] = 0

# Append the positive and negative example data frames
df = positive.append(negative, ignore_index=True)

# Drop duplicated negative examples
df.loc[:, 'duplicate'] = df.duplicated(keep=False)
df = df.loc[(df.duplicate == False) | (df.accident == 1), :].drop('duplicate', axis=1)

# Redefine the type of the segment identifier
df = df.astype({'segmentid':'uint64'})

# log
logging.info(f'Sampling of negative events done')
print(df.columns)

################################################################################
#                                                                              #
# This section of the script preprocesses the weather data.                    #
#                                                                              #
################################################################################

# Reshape the data frame
weather = pd.pivot_table(weather, values=weather.columns[(weather.columns != 'datetime') & (weather.columns != 'stationid')], index=['datetime'], columns=['stationid'])

# Smooth the weather conditions with an exponential moving average with a halflife of 12 hours
for c in weather.columns[~weather.columns.get_level_values(0).isin(['risky'])]:
    weather.loc[:, c] = weather.loc[:, c].interpolate(limit=12, limit_direction='both')
    missing = weather.loc[:, c].isna()
    weather.loc[:, c] = weather.loc[:, c].ewm(alpha=1-np.exp(-np.log(2)/12)).mean()
    weather.loc[missing, c] = np.nan

# Reset the index of the weather data frame
weather = weather.reset_index()

# log
logging.info(f'Weather preprocessing done')


################################################################################
#                                                                              #
# This section of the script matches road segments to weather conditions.      #
#                                                                              #
################################################################################

# Redefine the levels of the indices of the matched road traffic accidents and road segments data frame
df.columns = pd.MultiIndex.from_product([df.columns, ['']])

# Compute the inverse of the distance between each road segment and weather station
for station in range(locations.shape[0]):
    df.loc[:, ('inversedistance', locations.stationid.iloc[station])] = 1/df.geometry.distance(locations.geometry.iloc[station])

# Merge the two data frames
df = pd.merge(df, weather, how='left')

# Interpolate the weather variables to each road segment
for c in ['temperature', 'dewpoint', 'humidity', 'wdirection', 'wspeed', 'visibility', 'pressure', 'risky']:
    # Find the columns for each weather variable
    weathercolumns = df.columns[df.columns.get_level_values(0).isin([c])]

    # Find the weather stations for which the weather variable is recorded
    stations = weathercolumns.get_level_values(1).tolist()
    weathercolumns = weathercolumns.tolist()

    # Find the inverse distance columns for each of those weather stations
    geocolumns = [('inversedistance', station) for station in stations]

    # Compute a weighted average of each weather variable
    mask = df[weathercolumns].isna().to_numpy()
    weightedaverage = np.ma.average(np.ma.array(df[weathercolumns].to_numpy(), mask=mask), weights=np.ma.array(df[geocolumns].to_numpy(), mask=mask), axis=1)
    df.loc[:, (c, '')] = weightedaverage.data
    df.loc[weightedaverage.mask, (c, '')] = np.nan

    # Drop the weather columns for each station
    df = df.drop(weathercolumns, axis=1)

# Drop the inverse distance columns
stations = df.columns.get_level_values(1)[~df.columns.get_level_values(1).isin([''])].unique()
df = df.drop([('inversedistance', station) for station in stations], axis=1)

# Drop the first column index level
df.columns = df.columns.droplevel(level=1)

# log
logging.info(f'Matching of road segment to weather measurements done')
print(df.columns)

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
