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
                road_segments_ordered.segment_id as road_segment_id,
                ST_Distance(accidents_subset.geometry, road_segments_ordered.centroid) as distance_from_road_segment_centroid_in_m
            FROM
                accidents_subset
            LEFT JOIN LATERAL (
                SELECT
                    segment_id,
                    ST_Centroid(road_segments.geometry) as centroid
                FROM
                    road_segments
                WHERE
                    ST_Intersects(
                        ST_Buffer(accidents_subset.geometry, {MAX_DISTANCE_BETWEEN_ACCIDENT_AND_ROAD_SEGMENT_IN_METERS}),
                        road_segments.geometry)
                    = true
                ORDER BY
                    ST_Distance(accidents_subset.geometry, ST_Centroid(road_segments.geometry)) ASC
                LIMIT 1
            ) road_segments_ordered ON TRUE
        )
        SELECT
            DISTINCT ON (accident_id)
            accident_id,
            road_segment_id,
            distance_from_road_segment_centroid_in_m
        FROM
            accidents_potential_roadsegments
        ORDER BY
            accident_id,
            distance_from_road_segment_centroid_in_m ASC
    )
    SELECT * FROM accidents_roadsegments
    """

    # run query
    accidents_road_segments = pd.read_sql_query(
        sql=match_accidents_with_road_segments,
        con=engine
    )

    return accidents_road_segments


def match_accidents_with_weather_records(
        MAX_TIME_DIFF_BETWEEN_ACCIDENT_AND_WEATHER_RECORD_IN_SEC=7200,
        NBR_ACCIDENTS_IN_PROCESSED_BATCH=30
    ):

    match_query = f"""
    WITH accidents_weather_agg AS (
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
                        {NBR_ACCIDENTS_IN_PROCESSED_BATCH}
                )
                SELECT
                    accidents_subset.accident_id as accident_id,
                    accidents_subset.geometry as accident_geom,
                    weather_records_ordered.index as weather_record_index,
                    weather_records_ordered.station_id as weather_station_id,
                    weather_records_ordered.time_diff_in_s as time_diff_in_s
                FROM
                    accidents_subset
                LEFT JOIN LATERAL (
                    SELECT
                        index,
                        station_id,
                        ABS(EXTRACT(EPOCH FROM (accidents_subset.datetime::timestamp - weather_records.datetime::timestamp))) as time_diff_in_s
                    FROM
                        weather_records
                    WHERE
                        ABS(EXTRACT(EPOCH FROM (accidents_subset.datetime::timestamp - weather_records.datetime::timestamp))) <= {MAX_TIME_DIFF_BETWEEN_ACCIDENT_AND_WEATHER_RECORD_IN_SEC}
                    ORDER BY
                        ABS(EXTRACT(EPOCH FROM (accidents_subset.datetime::timestamp - weather_records.datetime::timestamp))) ASC
                    LIMIT 1
                ) weather_records_ordered ON TRUE
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
            accidents_weather.accident_id as accident_id,
            SUM(ST_Distance(weather_stations.geometry, accidents_weather.accident_geom)) as distance_sum_in_m,
            json_agg(
                json_build_object(
                    'station_id', accidents_weather.weather_station_id,
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
            accidents_weather.weather_station_id = weather_stations.station_id
        LEFT JOIN
            weather_records
        ON
            weather_records.index = accidents_weather.weather_record_index
        GROUP BY
            accidents_weather.accident_id
    )
    SELECT accident_id, distance_sum_in_m, weather_data FROM accidents_weather_agg
    """

    # run query
    accidents_weather_records = pd.read_sql_query(
        sql=match_query,
        con=engine
    )

    return accidents_weather_records


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
