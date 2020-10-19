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
        MAX_DISTANCE_BETWEEN_ACCIDENT_AND_ROAD_SEGMENT_IN_M=100,
        OFFSET=0
    ):
    """
        Function which pairs the accident with the nearest road segment
    """

    # build query
    match_accidents_with_road_segments = f"""
    WITH accidents_roadsegments AS (
        WITH accidents_subset AS (
            SELECT
                accident_id,
                geometry
            FROM
                accidents
            ORDER BY
                index
            OFFSET
                {OFFSET}
            LIMIT
                {NBR_ACCIDENTS_IN_PROCESSED_BATCH}
        )
        SELECT
            accidents_subset.accident_id as accident_id,
            road_segments_ordered.road_segment_id as road_segment_id,
            ST_Distance(accidents_subset.geometry, road_segments_ordered.geometry) as distance_from_road_segment_in_m
        FROM
            accidents_subset
        LEFT JOIN LATERAL (
            SELECT
                road_segment_id,
                geometry
            FROM
                road_segments
            ORDER BY
                ST_Distance(accidents_subset.geometry, road_segments.geometry) ASC
            OFFSET 0
            LIMIT 1
        ) road_segments_ordered ON TRUE
    )
    UPDATE
        accidents
    SET
        road_segment_id =
            CASE
                WHEN distance_from_road_segment_in_m <= {MAX_DISTANCE_BETWEEN_ACCIDENT_AND_ROAD_SEGMENT_IN_M}
                    THEN
                        accidents_roadsegments.road_segment_id
                ELSE
                    NULL
            END
    FROM
        accidents_roadsegments
    WHERE
        accidents.accident_id = accidents_roadsegments.accident_id
    """

    # run query
    with engine.connect() as connection:
        with connection.begin():
            connection.execute(match_accidents_with_road_segments)


def match_accidents_with_weather_records(
        NBR_ACCIDENTS_IN_PROCESSED_BATCH=30,
        OFFSET=0,
        MATCH_WITH_MAX_NBR_OF_WEATHER_STATIONS=3,
        WEATHER_STATION_MAX_DIST_FROM_ACCIDENT_IN_M=15000,
        WEATHER_DATA_MAX_TIME_DELTA_IN_S=7200
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
            ORDER BY
                index
            OFFSET
                {OFFSET}
            LIMIT
                {NBR_ACCIDENTS_IN_PROCESSED_BATCH}
        )
        SELECT
            accidents_subset.accident_id as accident_id,
            array_agg(weather_records_ordered.index) as weather_data
        FROM
            accidents_subset
        LEFT JOIN LATERAL (
            SELECT
                weather_records.index
            FROM
                weather_records
            LEFT JOIN
                weather_stations
            ON
                weather_stations.weather_station_id = weather_records.weather_station_id
            WHERE
                ST_Distance(weather_stations.geometry, accidents_subset.geometry) <= {WEATHER_STATION_MAX_DIST_FROM_ACCIDENT_IN_M}
                AND ABS(EXTRACT(EPOCH FROM (accidents_subset.datetime::timestamp - weather_records.datetime::timestamp))) <= {WEATHER_DATA_MAX_TIME_DELTA_IN_S}
            ORDER BY
                ABS(EXTRACT(EPOCH FROM (accidents_subset.datetime::timestamp - weather_records.datetime::timestamp))) ASC
            OFFSET 0
            LIMIT {MATCH_WITH_MAX_NBR_OF_WEATHER_STATIONS}
        ) weather_records_ordered ON TRUE
        GROUP BY
            accidents_subset.accident_id
    )
    UPDATE
        accidents
    SET
        weather_data = accidents_weather.weather_data
    FROM
        accidents_weather
    WHERE
        accidents.accident_id = accidents_weather.accident_id
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


def update_accident(accidents):

    accidents.to_sql(
        name='accidents',
        con=engine,
        if_exists='replace',
        index=False,

    )