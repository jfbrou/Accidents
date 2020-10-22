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
        (
            SELECT
                accidents_subset.accident_id as accident_id,
                road_segments_ordered.road_segment_id as road_segment_id,
                ST_Distance(accidents_subset.geometry, road_segments_ordered.geometry) as distance_from_road_segment_in_m
            FROM
                (
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
                ) AS accidents_subset
            LEFT JOIN LATERAL
                (
                    SELECT
                        road_segment_id,
                        geometry
                    FROM
                        road_segments
                    ORDER BY
                        ST_Distance(accidents_subset.geometry, road_segments.geometry) ASC
                    OFFSET 0
                    LIMIT 1
                ) AS road_segments_ordered
            ON TRUE
        ) AS accidents_roadsegments
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
        WEATHER_STATION_MAX_DIST_FROM_ACCIDENT_IN_M=15000,
        WEATHER_DATA_MAX_TIME_DELTA_IN_S=7200,
        MATCH_WITH_N_NEAREST_WEATHER_STATIONS=3
    ):


    match_query = f"""
    UPDATE
        accidents
    SET
        weather_data = accidents_weather.weather_data
    FROM
        (
            SELECT
                accidents_subset.accident_id as accident_id,
                array_agg(weather_records_ordered.index) as weather_data
            FROM
                (
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
                ) AS accidents_subset
            LEFT JOIN LATERAL
                (
                    SELECT
                        weather_records.index
                    FROM
                        weather_records
                    INNER JOIN
                        (
                            SELECT
                                weather_station_id
                            FROM
                                weather_stations
                            WHERE
                                ST_Distance(geometry, accidents_subset.geometry) <= {WEATHER_STATION_MAX_DIST_FROM_ACCIDENT_IN_M}
                            ORDER BY
                                ST_Distance(geometry, accidents_subset.geometry) ASC
                            OFFSET 0
                            LIMIT {MATCH_WITH_N_NEAREST_WEATHER_STATIONS}
                        ) AS weather_stations_subset
                    ON
                        weather_stations_subset.weather_station_id = weather_records.weather_station_id
                    WHERE
                        ABS(EXTRACT(EPOCH FROM (accidents_subset.datetime::timestamp - weather_records.datetime::timestamp))) <= {WEATHER_DATA_MAX_TIME_DELTA_IN_S}
                    ORDER BY
                        ABS(EXTRACT(EPOCH FROM (accidents_subset.datetime::timestamp - weather_records.datetime::timestamp))) ASC
                    OFFSET 0
                    LIMIT {MATCH_WITH_N_NEAREST_WEATHER_STATIONS}
                ) AS weather_records_ordered
            ON TRUE
            GROUP BY
                accidents_subset.accident_id
        ) AS accidents_weather
    WHERE
        accidents.accident_id = accidents_weather.accident_id
    """

    # run query
    with engine.connect() as connection:
        with connection.begin():
            connection.execute(match_query)


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


def get_accidents(
        OFFSET=0,
        LIMIT=1000
    ):
    """
        Returns the accidents with joined data
    """

    # SQL Query
    sql_query = f"""
        SELECT
            accidents.accident_id,
            accidents.datetime as accident_datetime,
            accidents.geometry as accident_geometry,
            road_segments.class as road_segment_class,
            road_segments.direction as road_segment_direction,
            road_segments.geometry as road_segment_geometry,
            accidents_weather_data_agg.temperature,
            accidents_weather_data_agg.dewpoint,
            accidents_weather_data_agg.humidity,
            accidents_weather_data_agg.wdirection,
            accidents_weather_data_agg.wspeed,
            accidents_weather_data_agg.visibility,
            accidents_weather_data_agg.pressure,
            accidents_weather_data_agg.risky
        FROM
            (
                WITH accidents_weather_data AS
                (
                    WITH accidents_subset AS (
                        SELECT
                            accident_id,
                            geometry,
                            weather_data
                        FROM
                            accidents
                        WHERE
                            weather_data IS NOT NULL
                        ORDER BY
                            index
                        OFFSET
                            {OFFSET}
                        LIMIT
                            {LIMIT}
                    )
                    SELECT
                        accidents_subset.accident_id as accident_id,
                        ST_Distance(accidents_subset.geometry, weather_stations.geometry) as distance_from_weather_station,
                        weather_records.temperature as temperature,
                        weather_records.dewpoint as dewpoint,
                        weather_records.humidity as humidity,
                        weather_records.wdirection as wdirection,
                        weather_records.wspeed as wspeed,
                        weather_records.visibility as visibility,
                        weather_records.pressure as pressure,
                        weather_records.risky as risky
                    FROM
                        accidents_subset,
                        unnest(weather_data) as weather_data_index
                    INNER JOIN
                        weather_records
                    ON
                        weather_data_index = weather_records.index
                    INNER JOIN
                        weather_stations
                    ON
                        weather_stations.weather_station_id = weather_records.weather_station_id
                )
                SELECT
                    accidents_weather_data.accident_id as accident_id,
                    SUM(accidents_weather_data.temperature*(1.0-(accidents_weather_data.distance_from_weather_station/grouped_accidents_weather_data.sum_of_distance_temperature))) as temperature,
                    SUM(accidents_weather_data.dewpoint*(1.0-(accidents_weather_data.distance_from_weather_station/grouped_accidents_weather_data.sum_of_distance_dewpoint))) as dewpoint,
                    SUM(accidents_weather_data.humidity*(1.0-(accidents_weather_data.distance_from_weather_station/grouped_accidents_weather_data.sum_of_distance_humidity))) as humidity,
                    SUM(accidents_weather_data.wdirection*(1.0-(accidents_weather_data.distance_from_weather_station/grouped_accidents_weather_data.sum_of_distance_wdirection))) as wdirection,
                    SUM(accidents_weather_data.wspeed*(1.0-(accidents_weather_data.distance_from_weather_station/grouped_accidents_weather_data.sum_of_distance_wspeed))) as wspeed,
                    SUM(accidents_weather_data.visibility*(1.0-(accidents_weather_data.distance_from_weather_station/grouped_accidents_weather_data.sum_of_distance_visibility))) as visibility,
                    SUM(accidents_weather_data.pressure*(1.0-(accidents_weather_data.distance_from_weather_station/grouped_accidents_weather_data.sum_of_distance_pressure))) as pressure,
                    SUM(accidents_weather_data.risky*(1.0-(accidents_weather_data.distance_from_weather_station/grouped_accidents_weather_data.sum_of_distance_risky))) as risky
                FROM
                    accidents_weather_data
                INNER JOIN
                    (
                        SELECT
                            accident_id,
                            SUM(
                                CASE
                                    WHEN temperature IS NULL THEN NULL
                                    WHEN temperature IS NOT NULL THEN distance_from_weather_station
                                END
                            ) as sum_of_distance_temperature,
                            SUM(
                                CASE
                                    WHEN dewpoint IS NULL THEN NULL
                                    WHEN dewpoint IS NOT NULL THEN distance_from_weather_station
                                END
                            ) as sum_of_distance_dewpoint,
                            SUM(
                                CASE
                                    WHEN humidity IS NULL THEN NULL
                                    WHEN humidity IS NOT NULL THEN distance_from_weather_station
                                END
                            ) as sum_of_distance_humidity,
                            SUM(
                                CASE
                                    WHEN wdirection IS NULL THEN NULL
                                    WHEN wdirection IS NOT NULL THEN distance_from_weather_station
                                END
                            ) as sum_of_distance_wdirection,
                            SUM(
                                CASE
                                    WHEN wspeed IS NULL THEN NULL
                                    WHEN wspeed IS NOT NULL THEN distance_from_weather_station
                                END
                            ) as sum_of_distance_wspeed,
                            SUM(
                                CASE
                                    WHEN visibility IS NULL THEN NULL
                                    WHEN visibility IS NOT NULL THEN distance_from_weather_station
                                END
                            ) as sum_of_distance_visibility,
                            SUM(
                                CASE
                                    WHEN pressure IS NULL THEN NULL
                                    WHEN pressure IS NOT NULL THEN distance_from_weather_station
                                END
                            ) as sum_of_distance_pressure,
                            SUM(
                                CASE
                                    WHEN risky IS NULL THEN NULL
                                    WHEN risky IS NOT NULL THEN distance_from_weather_station
                                END
                            ) as sum_of_distance_risky
                        FROM
                            accidents_weather_data
                        GROUP BY
                            accident_id
                    ) grouped_accidents_weather_data
                ON
                    grouped_accidents_weather_data.accident_id = accidents_weather_data.accident_id
                GROUP BY
                    accidents_weather_data.accident_id
            ) AS accidents_weather_data_agg
        INNER JOIN
            accidents
        ON
            accidents.accident_id = accidents_weather_data_agg.accident_id
        INNER JOIN
            road_segments
        ON
            road_segments.road_segment_id = accidents.road_segment_id
    """

    # run
    results = pd.read_sql_query(
        con=engine,
        sql=sql_query
    )

    return results
