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

    with engine.connect() as connection:
        with connection.begin():
            connection.execute(query)


def match_accidents_with_road_segments(
        NBR_ACCIDENTS_IN_PROCESSED_BATCH=30,
        OFFSET=0
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
        OFFSET
            {OFFSET}
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
        NBR_ACCIDENTS_IN_PROCESSED_BATCH=30,
        OFFSET=0
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
            OFFSET
                {OFFSET}
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


def sqlresults_to_dict(result_proxy):
    """
        Parses SQL Alchemy result proxy into a dict
    """

    # init var
    d = {}
    a = []

    # go through rows
    for row_proxy in result_proxy:

        # build up the dictionary
        for column, value in row_proxy.items():
            d = {**d, **{column: value}}

        # add to array
        a.append(d)

    return a


def get_accidents_count():
    """
        Returns the number of accidents
    """

    # SQL Query
    sql_query = 'SELECT COUNT(*) FROM accidents'

    count = 0
    with engine.connect() as connection:
        with connection.begin():

            # execute
            resultproxy = connection.execute(sql_query)

            # parse
            results = sqlresults_to_dict(resultproxy)

            # grab count
            count = results[0]['count']

    return count
