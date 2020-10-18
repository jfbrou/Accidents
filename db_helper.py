################################################################################
#                                                                              #
# Functions to interact with the database                                      #
#                                                                              #
################################################################################

import numpy as np
import pandas as pd
import geopandas as gpd

from sqlalchemy import create_engine

# Set up database connection engine
from config import username, password, hostname, port, db_name
engine = create_engine(
    f'postgresql+psycopg2://{username}:{password}@{hostname}:{port}/{db_name}',
    connect_args={'options': '-csearch_path={}'.format('public')}
)

def execute_query(query):
    """
        Executes a SQL query on the database
    """
    try:
        with engine.connect() as connection:
            with connection.begin():
                connection.execute(query)
    except:
        return False

    return True


def match_accidents_with_road_segments(
        NBR_ACCIDENTS_IN_PROCESSED_BATCH=30
    ):
    """
        Function which pairs the accident with the nearest road segment
    """

    # build query
    match_accidents_with_road_segments = f"""
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
        road_segments_ordered.road_segment_id as road_segment_id,
        ST_Distance(accidents_subset.geometry, road_segments_ordered.centroid) as distance_from_road_segment_centroid_in_m
    FROM
        accidents_subset
    LEFT JOIN LATERAL (
        SELECT
            road_segment_id,
            ST_Centroid(road_segments.geometry) as centroid
        FROM
            road_segments
        ORDER BY
            ST_Distance(accidents_subset.geometry, ST_Centroid(road_segments.geometry)) ASC
        LIMIT 1
    ) road_segments_ordered ON TRUE
    """

    # run query
    accidents_road_segments = pd.read_sql_query(
        sql=match_accidents_with_road_segments,
        con=engine
    )

    return accidents_road_segments


def match_accidents_with_weather_records(
        NBR_ACCIDENTS_IN_PROCESSED_BATCH=30
    ):

    match_query = f"""
    WITH accidents_weather AS (
        WITH accidents_subset AS (
            SELECT
                accident_id,
                geometry,
                datetime
            FROM
                accidents
            WHERE
                weather_data IS NULL
            LIMIT
                {NBR_ACCIDENTS_IN_PROCESSED_BATCH}
        )
        SELECT
            accidents_subset.accident_id as accident_id,
            accidents_subset.geometry as accident_geom,
            weather_records_ordered.index as weather_record_index,
            weather_records_ordered.weather_station_id as weather_station_id,
            weather_records_ordered.time_diff_in_s as time_diff_in_s
        FROM
            accidents_subset
        LEFT JOIN LATERAL (
            SELECT
                index,
                weather_station_id,
                ABS(EXTRACT(EPOCH FROM (accidents_subset.datetime::timestamp - weather_records.datetime::timestamp))) as time_diff_in_s
            FROM
                weather_records
            ORDER BY
                ABS(EXTRACT(EPOCH FROM (accidents_subset.datetime::timestamp - weather_records.datetime::timestamp))) ASC
            LIMIT 1
        ) weather_records_ordered ON TRUE
    )
    SELECT
        accidents_weather.accident_id as accident_id,
        SUM(ST_Distance(weather_stations.geometry, accidents_weather.accident_geom)) as distance_sum_in_m,
        json_agg(
            json_build_object(
                'weather_station_id', accidents_weather.weather_station_id,
                'time_diff_in_s', accidents_weather.time_diff_in_s,
                'distance_in_m', ST_Distance(weather_stations.geometry, accidents_weather.accident_geom),
                'temperature', weather_records.temperature,
                'dewpoint', weather_records.dewpoint,
                'humidity',weather_records.humidity,
                'wdirection',weather_records.wdirection,
                'wspeed',weather_records.wspeed,
                'visibility',weather_records.visibility,
                'pressure',weather_records.pressure,
                'risky',weather_records.risky
            )
        ) as weather_data
    FROM
        accidents_weather
    LEFT JOIN
        weather_stations
    ON
        accidents_weather.weather_station_id = weather_stations.weather_station_id
    LEFT JOIN
        weather_records
    ON
        weather_records.index = accidents_weather.weather_record_index
    GROUP BY
        accidents_weather.accident_id
    """

    # run query
    accidents_weather_records = pd.read_sql_query(
        sql=match_query,
        con=engine
    )

    return accidents_weather_records



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
