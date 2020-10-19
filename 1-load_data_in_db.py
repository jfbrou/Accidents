################################################################################
#                                                                              #
# This script loads 'weather.csv', 'accidents.csv' and 'segments.zip'          #
# into the PostgreSQL database after doing some preprocessing                  #
#                                                                              #
################################################################################


# Import libraries
import os
import datetime

import pandas as pd
import numpy as np
import geopandas as gpd

from sqlalchemy import create_engine, types
from sqlalchemy.dialects import postgresql as postgresTypes
from geoalchemy2.types import Geometry as geoTypes

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

# Create a folder that contains all data files
if os.path.isdir(os.path.join(path, 'Data')) == False:
    raise Exception('Data directory does not exist, run retrieve script')
data_dir_path = os.path.join(path, 'Data')


def execute_query(query):
    """
        Executes a SQL query on the database
    """

    with engine.connect() as connection:
        with connection.begin():
            connection.execute(query)



################################################################################
#                                                                              #
#                           create the tables                                  #
#                                                                              #
################################################################################

# drop tables
for d in ['accidents', 'road_segments', 'weather_stations', 'weather_records']:
    drop_query = f'DROP TABLE IF EXISTS {d} CASCADE'
    execute_query(drop_query)

# create tables queries
create_table_road_segments_query = f"""
        CREATE TABLE road_segments (
            index INT GENERATED ALWAYS AS IDENTITY,
            road_segment_id INT NOT NULL UNIQUE,
            class INT,
            direction INT,
            geometry geometry(LINESTRING,32188) NOT NULL,
            PRIMARY KEY(index)
        )
        """

create_table_weather_stations_query = f"""
        CREATE TABLE weather_stations (
            index INT GENERATED ALWAYS AS IDENTITY,
            weather_station_id VARCHAR(32) NOT NULL UNIQUE,
            geometry geometry(POINT,32188) NOT NULL,
            PRIMARY KEY(index)
        )
        """

create_table_weather_records_query = f"""
        CREATE TABLE weather_records (
            index BIGINT GENERATED ALWAYS AS IDENTITY,
            weather_station_id VARCHAR(32) NOT NULL,
            datetime TIMESTAMP NOT NULL,
            temperature FLOAT8,
            dewpoint FLOAT8,
            humidity FLOAT8,
            wdirection FLOAT8,
            wspeed FLOAT8,
            visibility FLOAT8,
            pressure FLOAT8,
            risky FLOAT8,
            PRIMARY KEY(index),
            FOREIGN KEY (weather_station_id) REFERENCES weather_stations (weather_station_id)
        )
        """

create_table_accidents_query = f"""
        CREATE TABLE accidents (
            index INT GENERATED ALWAYS AS IDENTITY,
            accident_id VARCHAR(64) NOT NULL UNIQUE,
            datetime TIMESTAMP NOT NULL,
            geometry geometry(POINT,32188) NOT NULL,
            road_segment_id INT,
            weather_data BIGINT[],
            PRIMARY KEY(index),
            FOREIGN KEY (road_segment_id) REFERENCES road_segments (road_segment_id)
        )
        """

# execute
execute_query(create_table_road_segments_query)
execute_query(create_table_weather_stations_query)
execute_query(create_table_weather_records_query)
execute_query(create_table_accidents_query)


################################################################################
#                                                                              #
#                              weather stations                                #
#                                                                              #
################################################################################

# Define the variable types
columns = ['Longitude (x)', 'Latitude (y)', 'Climate ID', 'Date/Time', 'Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)', 'Wind Dir (10s deg)', 'Wind Spd (km/h)', 'Visibility (km)', 'Stn Press (kPa)', 'Weather']
dtypes = dict(zip(columns, ['float64', 'float64', 'object', 'object', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'object']))

# Load the data
weather = pd.read_csv(os.path.join(data_dir_path, 'weather.csv'), dtype=dtypes)

# Rename columns
weather = weather.rename(columns=dict(zip(weather.columns, ['longitude', 'latitude', 'weather_station_id', 'datetime', 'temperature', 'dewpoint', 'humidity', 'wdirection', 'wspeed', 'visibility', 'pressure', 'risky'])))

# Redefine the type of the station identifier
weather.loc[:, 'weather_station_id'] = weather.weather_station_id.astype('str')


# Get the unique stations
stations = weather.drop_duplicates(['weather_station_id'])

# remove columns
stations = stations[['weather_station_id', 'latitude', 'longitude']]

# Convert the data frame to a geo data frame
stations = gpd.GeoDataFrame(stations, crs='EPSG:4326', geometry=gpd.points_from_xy(stations.longitude, stations.latitude)).to_crs('EPSG:32188')

# Drop the geometry column
weather = weather.drop(['latitude', 'longitude'], axis=1)
stations = stations.drop(['longitude', 'latitude'], axis=1)

# Reset the data frame's index
stations = stations.reset_index(drop=True)

# add to database
stations.to_postgis(
    con=engine,
    name='weather_stations',
    if_exists='append',
    dtype={
        'weather_station_id': types.Text(),
        'geometry': geoTypes(geometry_type='POINT', srid=32188)
    }
)

# log
logging.info(f'Stations loaded')


################################################################################
#                                                                              #
#                               weather data                                   #
#                                                                              #
################################################################################

# Redefine the types of the date and time columns
weather.loc[:, 'datetime'] = pd.to_datetime(weather.datetime, infer_datetime_format=True)

# Localize the time zone
weather.loc[:, 'datetime'] = weather.datetime.dt.tz_localize('EST')
weather.loc[:, 'datetime'] = weather.datetime.dt.tz_convert(None)

# Convert the risky weather categorical variable to a binary variable
weather.loc[:, 'risky'] = (weather.risky.notna() & (weather.risky != 'Mainly Clear') & (weather.risky != 'Clear')).astype('int64')

# Reset the index of the weather data frame
weather = weather.reset_index(drop=True)

# add to db
weather.to_sql(
    con=engine,
    name='weather_records',
    if_exists='append',
    index=False,
    dtype={
        'weather_station_id': types.Text(),
        'datetime': types.DateTime(),
        'temperature': types.Float(),
        'dewpoint': types.Float(),
        'humidity': types.Float(),
        'wdirection': types.Float(),
        'wspeed': types.Float(),
        'visibility': types.Float(),
        'pressure': types.Float(),
        'risky': types.Integer()
    }
)

# log
logging.info(f'Weather loaded')



################################################################################
#                                                                              #
#                               segments data                                  #
#                                                                              #
################################################################################

# Load the data
segments = gpd.read_file(r'zip://'+os.path.join(data_dir_path, 'segments.zip'))

# Keep relevant columns
segments = segments.loc[:, ['ID_TRC', 'CLASSE', 'SENS_CIR', 'geometry']]

# Rename columns
segments = segments.rename(columns={'ID_TRC':'road_segment_id', 'CLASSE':'class', 'SENS_CIR':'direction'})

# Redefine the types of each column
segments = segments.astype({'road_segment_id':'int64', 'class':'int64', 'direction':'int64'})

# Reset the data frame's index
segments = segments.reset_index(drop=True)

# add to db
segments.to_postgis(
    con=engine,
    name='road_segments',
    if_exists='append',
    dtype={
        'road_segment_id': types.Integer(),
        'class': types.Integer(),
        'direction': types.Integer(),
        'geometry': geoTypes(geometry_type='LINESTRING', srid=32188)
    }
)

# log
logging.info(f'Segments loaded')




################################################################################
#                                                                              #
#                        traffic accidents data                                #
#                                                                              #
################################################################################

# Load the data
accidents = pd.read_csv(os.path.join(data_dir_path, 'accidents.csv'), usecols=['NO_SEQ_COLL', 'DT_ACCDN', 'HEURE_ACCDN', 'LOC_X', 'LOC_Y'])

# Rename columns
accidents = accidents.rename(columns={'NO_SEQ_COLL':'accident_id', 'DT_ACCDN':'date', 'HEURE_ACCDN':'time', 'LOC_X':'longitude', 'LOC_Y':'latitude'})

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

# Convert the lat/lng to a point using geo data
accidents = gpd.GeoDataFrame(accidents, crs='EPSG:32188', geometry=gpd.points_from_xy(accidents.longitude, accidents.latitude))
accidents = accidents.drop(['longitude', 'latitude'], axis=1)

# Add missing columns
accidents['road_segment_id'] = None
accidents['weather_data'] = None

# Reset the data frame's index
accidents = accidents.reset_index(drop=True)

# add to database
accidents.to_postgis(
    con=engine,
    name='accidents',
    if_exists='append',
    dtype={
        'accident_id': types.Text(),
        'datetime': types.DateTime(),
        'geometry': geoTypes(geometry_type='POINT', srid=32188),
        'road_segment_id': types.Integer(),
        'weather_data': postgresTypes.JSONB
    }
)

# log
logging.info(f'Accidents loaded')
