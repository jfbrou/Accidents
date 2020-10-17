################################################################################
#                                                                              #
# This section of the script imports libraries, creates directories and        #
# retrieves the road traffic accidents, road segments and weather data.        #
#                                                                              #
################################################################################

# Import libraries
import os
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import geopandas as gpd
from sqlalchemy import create_engine

#from pvlib import solarposition


# Set up database connection engine
from config import username, password, hostname, port, db_name
engine = create_engine(
    f'postgresql+psycopg2://{username}:{password}@{hostname}:{port}/{db_name}',
    connect_args={'options': '-csearch_path={}'.format('public')}
)

# Logging
import logging
logging.basicConfig(level=logging.INFO)

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


def match_accidents_with_road_segments(
        MAX_DISTANCE_BETWEEN_ACCIDENT_AND_ROAD_SEGMENT_IN_METERS=100,
        NBR_ACCIDENTS_IN_PROCESSED_BATCH=30
    ):
    """
        Function which pairs the accident with the nearest road segment
    """

    # build query
    match_accidents_with_road_segments = f"""
    WITH accidents_roadsegments AS (
        WITH accidents_potential_roadsegments AS (
            WITH accidents_subset AS (
                SELECT
                    accident_id,
                    geometry
                FROM
                    accidents
                WHERE
                    road_segment_id IS NULL
                LIMIT
                    {NBR_ACCIDENTS_IN_PROCESSED_BATCH}
            )
            SELECT
                accidents_subset.accident_id as accident_id,
                accidents_subset.geometry as accident_geom,
                road_segments.segment_id as road_segment_id,
                ST_Distance(accidents_subset.geometry, ST_Centroid(road_segments.geometry)) as distance
            FROM
                accidents_subset,
                road_segments
            WHERE
                ST_Intersects(
                    ST_Buffer(accidents_subset.geometry, {MAX_DISTANCE_BETWEEN_ACCIDENT_AND_ROAD_SEGMENT_IN_METERS}),
                    road_segments.geometry)
                = true
        )
        SELECT
            DISTINCT ON (accident_id)
            accident_id,
            road_segment_id,
            accident_geom,
            distance
        FROM
            accidents_potential_roadsegments
        ORDER BY
            accident_id,
            distance ASC
    )
    UPDATE
        accidents
    SET
        road_segment_id = accidents_roadsegments.road_segment_id
    FROM
        accidents_roadsegments
    WHERE
        accidents.accident_id = accidents_roadsegments.accident_id
    """

    # run query
    success = True

    try:
        with engine.connect() as connection:
            with connection.begin():
                connection.execute(match_accidents_with_road_segments)
    except:
        success = False

    return success


def match_accidents_with_weathers():

    query = f"""
    WITH accidents_weather_full AS (
        WITH accidents_weather AS (
            WITH accidents_potential_weather AS (
                WITH accidents_subset AS (
                    SELECT
                        accident_id,
                        geometry,
                        datetime
                    FROM
                        accidents
                    WHERE
                        weather IS NULL
                    LIMIT
                        2
                )
                SELECT
                    accidents_subset.accident_id as accident_id,
                    accidents_subset.geometry as accident_geom,
                    weather_records.index as weather_record_index,
                    weather_records.station_id as weather_station_id,
                    ABS(EXTRACT(EPOCH FROM (accidents_subset.datetime::timestamp - weather_records.datetime::timestamp))) as time_diff_in_s
                FROM
                    accidents_subset,
                    weather_records
                WHERE
                    ABS(EXTRACT(EPOCH FROM (accidents_subset.datetime::timestamp - weather_records.datetime::timestamp))) <= 3600
            )
            SELECT
                DISTINCT ON (accident_id, weather_station_id)
                accident_id,
                accident_geom,
                weather_record_index,
                weather_station_id,
                time_diff_in_s
            FROM
                accidents_potential_weather
            ORDER BY
                accident_id,
                weather_station_id,
                time_diff_in_s ASC
        )
        SELECT
            accidents_weather.accident_id,
            accidents_weather.weather_record_index,
            accidents_weather.weather_station_id,
            accidents_weather.time_diff_in_s,
            ST_Distance(weather_stations.geometry, accidents_weather.accident_geom) AS distance_in_m,
            weather_records.temperature,
            weather_records.dewpoint,
            weather_records.humidity,
            weather_records.wdirection,
            weather_records.wspeed,
            weather_records.visibility,
            weather_records.pressure,
            weather_records.risky
        FROM
            accidents_weather
        LEFT JOIN
            weather_stations
        ON
            accidents_weather.weather_station_id = weather_stations.station_id
        LEFT JOIN
            weather_records
        ON
            weather_records.index = accidents_weather.weather_record_index
    )
    SELECT * FROM accidents_weather_full;
    """


def match_accident_with_road_segment(accident_id):
    """
        Function which pairs the accident with the nearest road segment
    """

    # Find the nearest road segment
    nearest_road_segment_query = f"""
        WITH distances AS (
            SELECT
                accidents.accident_id as accident_id,
                road_segments.segment_id as road_segment_id,
                ST_Distance(accidents.geometry, ST_Centroid(road_segments.geometry)) as distance_in_m
            FROM
                accidents,
                road_segments
            WHERE
                accidents.accident_id = '{accident_id}'
        )
        SELECT DISTINCT ON (accident_id)
            accident_id,
            road_segment_id,
            distance_in_m
        FROM
            distances
        ORDER BY
            accident_id, distance_in_m ASC
    """

    nearest_road_segment = pd.read_sql_query(
        sql=nearest_road_segment_query,
        con=engine
    )

    return nearest_road_segment


def match_accident_with_weather_data(accident_id):
    """
        Function which pairs the accident with the nearest weather station
    """

    # Get weather data for accident
    weather_query = f"""
        WITH acci_weather AS (
            WITH acci_weather_diff AS (
                SELECT
                    weather_records.index as weather_record_index,
                    weather_records.station_id as weather_station_id,
                    ABS(EXTRACT(EPOCH FROM (accident.datetime::timestamp - weather_records.datetime::timestamp))) as time_diff_in_s
                FROM
                    weather_records, (SELECT datetime FROM accidents WHERE accident_id = '{accident_id}') accident
            )
            SELECT DISTINCT ON (weather_station_id)
                weather_record_index,
                weather_station_id,
                time_diff_in_s FROM acci_weather_diff
            ORDER BY
                weather_station_id,
                time_diff_in_s ASC
        )
        SELECT
            acci_weather.weather_record_index as weather_record_index,
            acci_weather.weather_station_id as weather_station_id,
            acci_weather.time_diff_in_s as time_diff_in_s,
            ST_Distance(weather_stations.geometry, accidents.geometry) as distance_diff_in_m
        FROM acci_weather
        LEFT JOIN weather_stations ON acci_weather.weather_station_id = weather_stations.station_id
        LEFT JOIN accidents ON accidents.accident_id = '{accident_id}'
    """

    weather_data = pd.read_sql_query(
        sql=weather_query,
        con=engine
    )

    return weather_data


def get_accidents(LIMIT=None):
    """
        Loads the accidents as a pandas dataframe from the database
    """

    # SQL Query
    sql_query = 'SELECT * FROM accidents'
    if(LIMIT is not None and type(LIMIT) == int):
        sql_query = sql_query + f' LIMIT {LIMIT}'

    # Pull the data from the database
    accidents = gpd.read_postgis(
        sql=sql_query,
        con=engine,
        geom_col='geometry'
    )

    # log
    logging.info(f'Accidents loaded')

    return accidents


def get_weather_records():
    """
        Loads the weather_records as a pandas dataframe from the database
    """

    # SQL Query
    sql_query = 'SELECT * FROM weather_records'

    # Pull the data from the database
    weather_records = pd.read_sql_query(
        sql=sql_query,
        con=engine
    )

    # log
    logging.info(f'Weather records loaded')

    return weather_records


def get_weather_stations():
    """
        Loads the weather_stations as a pandas dataframe from the database
    """

    # SQL Query
    sql_query = 'SELECT * FROM weather_stations'

    # Pull the data from the database
    weather_stations = gpd.read_postgis(
        sql=sql_query,
        con=engine,
        geom_col='geometry'
    )

    # log
    logging.info(f'Weather stations loaded')

    return weather_stations


def get_road_segments():
    """
        Loads the road_segments as a pandas dataframe from the database
    """

    # SQL Query
    sql_query = 'SELECT * FROM road_segments'

    # Pull the data from the database
    segments = gpd.read_postgis(
        sql=sql_query,
        con=engine,
        geom_col='geometry'
    )

    # log
    logging.info(f'Road segments loaded')

    return segments


def get_weighted_weather(weather_stations_data):

    # convert to dict
    weather_records = weather_stations_data.to_dict(orient='records')

    # accumulate distance
    total_distance = 0.0
    for weather_record in weather_records:
        total_distance += weather_record['distance_diff_in_m']

    # weighted average
    for i, weather_record in enumerate(weather_records):
        weather_records[i]['dist_weighted'] = 1/(weather_record['distance_diff_in_m']/total_distance)

    for weather_record in weather_records:
        print(weather_record['dist_weighted'])

    print('\n\n')


################################################################################
#                                                                              #
#                        Loading the data from the DB                          #
#                                                                              #
################################################################################

res = match_accidents_with_road_segments(
    MAX_DISTANCE_BETWEEN_ACCIDENT_AND_ROAD_SEGMENT_IN_METERS=50,
    NBR_ACCIDENTS_IN_PROCESSED_BATCH=1000
)

print(res)
raise Exception('stop')

#accidents = get_accidents(LIMIT=100)

# road_segments = get_road_segments()


################################################################################
#                                                                              #
#                                 Produce PDFs                                 #
#                                                                              #
################################################################################

# # write
# out_path = os.path.join(figures_dir_path, 'accidents_2019.pdf')
# draw_as_pdf(accidents, out_path)

# # write
# out_path = os.path.join(figures_dir_path, 'segments.pdf')
# draw_as_pdf(road_segments, out_path)

for index, row in accidents.iterrows():

    if(index > 5):
        break

    # grab id
    accident_id = row['accident_id']

    # link with road segment and weather
    segment_data = match_accident_with_road_segment(accident_id)
    weather_data = match_accident_with_weather_data(accident_id)

    # process
    get_weighted_weather(weather_data)



raise Exception('stop')



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
