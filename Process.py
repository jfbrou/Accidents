################################################################################
#                                                                              #
# This section of the script imports libraries, creates directories and        #
# retrieves the road traffic accidents, road segments and weather data.        #
#                                                                              #
################################################################################

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, LineString
from pvlib import solarposition
import os
import urllib.request as urllib
import datetime

# Find the current working directory
path = os.getcwd()

# Create a folder that contains all data files
if os.path.isdir(os.path.join(path, 'Data')) == False:
    os.mkdir('Data')
data = os.path.join(path, 'Data')

# Create a folder that contains all figures
if os.path.isdir(os.path.join(path, 'Figures')) == False:
    os.mkdir('Figures')
figures = os.path.join(path, 'Figures')

# Retrieve the road traffic accidents, road segments and weather data
%run -i "Retrieve.py"

################################################################################
#                                                                              #
# This section of the script preprocesses the road traffic accidents data.     #
#                                                                              #
################################################################################

# Load the data
accidents = pd.read_csv(os.path.join(data, 'accidents.csv'), usecols=['NO_SEQ_COLL', 'DT_ACCDN', 'HEURE_ACCDN', 'LOC_X', 'LOC_Y'])

# Rename columns
accidents = accidents.rename(columns={'NO_SEQ_COLL':'accidentid', 'DT_ACCDN':'date', 'HEURE_ACCDN':'time', 'LOC_X':'longitude', 'LOC_Y':'latitude'})

# Drop observations for which we do not observe the time or location of the event
accidents = accidents.loc[accidents.longitude.notna() & accidents.latitude.notna() & (accidents.time != 'Non précisé'), :]

# Redefine the types of the date and time columns
accidents.loc[:, 'date'] = pd.to_datetime(accidents.date, infer_datetime_format=True)
accidents.loc[:, 'time'] = pd.to_datetime(accidents.time.str[:8], infer_datetime_format=True).dt.hour
accidents.loc[:, 'datetime'] = accidents.date+accidents.time.transform(lambda x: datetime.timedelta(hours=x))
accidents = accidents.drop(['date', 'time'], axis=1)

# Localize the time zone
accidents.loc[:, 'datetime'] = accidents.datetime.dt.tz_localize('US/Eastern', ambiguous=True, nonexistent='shift_forward')
accidents.loc[:, 'datetime'] = accidents.datetime.dt.tz_convert(None)

# Convert the data frame to a geo data frame
accidents = gpd.GeoDataFrame(accidents, crs='EPSG:32188', geometry=gpd.points_from_xy(accidents.longitude, accidents.latitude))
accidents = accidents.drop(['longitude', 'latitude'], axis=1)

# Reset the data frame's index
accidents = accidents.reset_index(drop=True)

# Plot all road traffic accidents in 2019
fig, ax = plt.subplots()
accidents.plot(ax=ax, color='teal', markersize=0.1)
ax.set_axis_off()
fig.tight_layout()
fig.savefig(os.path.join(figures, 'accidents.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script preprocesses the road segments data.              #
#                                                                              #
################################################################################

# Load the data
segments = gpd.read_file(r'zip://'+os.path.join(data, 'segments.zip'))

# Keep relevant columns
segments = segments.loc[:, ['ID_TRC', 'CLASSE', 'SENS_CIR', 'geometry']]

# Rename columns
segments = segments.rename(columns={'ID_TRC':'segmentid', 'CLASSE':'class', 'SENS_CIR':'direction'})

# Find the number of intersections for each segment
segments.loc[:, 'intersections'] = gpd.sjoin(segments, segments.loc[:, ['geometry']], how='left', op='intersects').groupby('segmentid', as_index=False).agg({'index_right':'count'}).index_right

# Redefine the types of each column
segments = segments.astype({'segmentid':'int64', 'class':'int64', 'direction':'int64', 'intersections':'int64'})

# Reset the data frame's index
segments = segments.reset_index(drop=True)

# Plot all road segments
fig, ax = plt.subplots()
segments.plot(ax=ax, edgecolor='teal', linewidth=0.1)
ax.set_axis_off()
fig.tight_layout()
fig.savefig(os.path.join(figures, 'segments.pdf'), format='pdf')
plt.close()

################################################################################
#                                                                              #
# This section of the script matches road traffic accidents to road segments.  #
#                                                                              #
################################################################################

# Create a data frame for matched road traffic accidents and road segments
match = accidents.copy(deep=True)

# Define a 100 meter radius circle around the location of each accident
match.loc[:, 'geometry'] = match.geometry.buffer(100)

# Find all road traffic accidents 100 meter radiuses that intersect with a road segment
match = gpd.sjoin(match, segments.loc[:, ['segmentid', 'geometry']], how='left', op='intersects')
match.loc[:, 'geometry'] = match.geometry.centroid

# Drop accidents that were not matched to any road segment
match = match.loc[match.index_right.notna(), :].drop('index_right', axis=1)

# Create a data frame with the geometry of matched road segments
df = gpd.GeoDataFrame(pd.merge(match.loc[:, 'segmentid'], segments.loc[:, ['segmentid', 'geometry']], how='left'), geometry='geometry')

# Reset the index of the matched road traffic accidents and road segments data frame
match = match.reset_index(drop=True)

# Compute the distance between the matched accidents and segments
match.loc[:, 'distance'] = match.distance(df)

# For each accident, keep the match with the smallest distance
match = pd.merge(match, match.groupby('accidentid', as_index=False).agg({'distance':'idxmin'}).rename(columns={'distance':'smallest'}), how='left')
match = match.loc[match.index == match.smallest, :].drop(['geometry', 'distance', 'smallest'], axis=1)

# Merge the matched data frame with the segments data frame
positive = gpd.GeoDataFrame(pd.merge(match, segments, how='left'), geometry='geometry').drop('accidentid', axis=1)

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

################################################################################
#                                                                              #
# This section of the script preprocesses the weather data.                    #
#                                                                              #
################################################################################

# Define the variable types
columns = ['Longitude (x)', 'Latitude (y)', 'Climate ID', 'Date/Time', 'Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)', 'Wind Dir (10s deg)', 'Wind Spd (km/h)', 'Visibility (km)', 'Stn Press (kPa)', 'Weather']
types = dict(zip(columns, ['float64', 'float64', 'object', 'object', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'object']))

# Load the data
weather = pd.read_csv(os.path.join(data, 'weather.csv'), dtype=types)

# Rename columns
weather = weather.rename(columns=dict(zip(weather.columns, ['longitude', 'latitude', 'stationid', 'datetime', 'temperature', 'dewpoint', 'humidity', 'wdirection', 'wspeed', 'visibility', 'pressure', 'risky'])))

# Redefine the types of the date and time columns
weather.loc[:, 'datetime'] = pd.to_datetime(weather.datetime, infer_datetime_format=True)

# Localize the time zone
weather.loc[:, 'datetime'] = weather.datetime.dt.tz_localize('EST')
weather.loc[:, 'datetime'] = weather.datetime.dt.tz_convert(None)

# Redefine the type of the station identifier
weather.loc[:, 'stationid'] = weather.stationid.astype('str')

# Convert the data frame to a geo data frame
weather = gpd.GeoDataFrame(weather, crs='EPSG:4326', geometry=gpd.points_from_xy(weather.longitude, weather.latitude)).to_crs('EPSG:32188')
weather = weather.drop(['longitude', 'latitude'], axis=1)

# Record the stations' location
locations = weather.groupby('stationid', as_index=False).agg({'geometry':'first'})

# Drop the geometry column
weather = weather.drop('geometry', axis=1)

# Convert the risky weather categorical variable to a binary variable
weather.loc[:, 'risky'] = (weather.risky.notna() & (weather.risky != 'Mainly Clear') & (weather.risky != 'Clear')).astype('int64')

# Reshape the data frame
weather = pd.pivot_table(weather, values=weather.columns[(weather.columns != 'datetime') & (weather.columns != 'stationid')], index=['datetime'], columns=['stationid'])

# Reset the index of the weather data frame
weather = weather.reset_index()

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
binary = pd.get_dummies(df.loc[:, 'class']).rename(columns=names)
df = pd.merge(df, binary, left_index=True, right_index=True)
df = df.drop('class', axis=1)

# Convert the road segment direction to binary variables
names = dict(zip(df.loc[:, 'direction'].unique().tolist(), ['direction'+str(x) for x in df.loc[:, 'direction'].unique().tolist()]))
binary = pd.get_dummies(df.loc[:, 'direction']).rename(columns=names)
df = pd.merge(df, binary, left_index=True, right_index=True)
df = df.drop('direction', axis=1)

# Compute the road segment length
df.loc[:, 'roadlength'] = df.geometry.length

# Compute the area of the road segment's convex hull
df.loc[:, 'convexhull'] = df.geometry.convex_hull.area

# Compute the road segment sinuosity
df.loc[:, 'sinuosity'] = df.geometry.apply(lambda x: x.length/LineString((x.coords[0], x.coords[-1])).length)

# Convert months to binary variables
names = dict(zip(df.datetime.dt.month.unique().tolist(), ['month'+str(x) for x in df.datetime.dt.month.unique().tolist()]))
binary = pd.get_dummies(df.datetime.dt.month).rename(columns=names)
df = pd.merge(df, binary, left_index=True, right_index=True)

# Convert weeks to binary variables
names = dict(zip(list(df.datetime.dt.isocalendar().week.unique()), ['week'+str(x) for x in list(df.datetime.dt.isocalendar().week.unique())]))
binary = pd.get_dummies(df.datetime.dt.isocalendar().week).rename(columns=names)
df = pd.merge(df, binary, left_index=True, right_index=True)

# Convert days to binary variables
names = dict(zip(df.datetime.dt.day.unique().tolist(), ['day'+str(x) for x in df.datetime.dt.day.unique().tolist()]))
binary = pd.get_dummies(df.datetime.dt.day).rename(columns=names)
df = pd.merge(df, binary, left_index=True, right_index=True)

# Convert weekdays to binary variables
names = dict(zip(df.datetime.dt.weekday.unique().tolist(), ['weekday'+str(x) for x in df.datetime.dt.weekday.unique().tolist()]))
binary = pd.get_dummies(df.datetime.dt.weekday).rename(columns=names)
df = pd.merge(df, binary, left_index=True, right_index=True)

# Convert hours to binary variables
names = dict(zip(df.datetime.dt.hour.unique().tolist(), ['hour'+str(x) for x in df.datetime.dt.hour.unique().tolist()]))
binary = pd.get_dummies(df.datetime.dt.hour).rename(columns=names)
df = pd.merge(df, binary, left_index=True, right_index=True)

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
