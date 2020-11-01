################################################################################
#                                                                              #
# Match the accidents with the road segments and weather records as a CRON job #
#                                                                              #
################################################################################

# Import libraries
import os
import time
import threading

# Logging
import logging
logging.basicConfig(level=logging.INFO)

# Import database helper
import db_helper

# get accidents count
accidents_count = db_helper.get_accidents_count()

# processing start index
road_segments_process_start_index = 104300
weather_records_process_start_index = 60000

def match_with_road_segments():

    # nbr of accidents per process batch
    nbr_accidents_in_batch = 1000

    # init index keeping track of what we have processed
    process_index = road_segments_process_start_index

    while(process_index < accidents_count):

        # get accidents matched with road segments
        db_helper.match_accidents_with_road_segments(
            NBR_ACCIDENTS_IN_PROCESSED_BATCH=nbr_accidents_in_batch,
            MAX_DISTANCE_BETWEEN_ACCIDENT_AND_ROAD_SEGMENT_IN_M=25,
            OFFSET=process_index
        )

        # prompt
        logging.info(f'Accidents matched with road segments from index {process_index} to {process_index+nbr_accidents_in_batch}')

        # update
        process_index += nbr_accidents_in_batch

        # Wait
        time.sleep(5)


def match_with_weather_records():

    # nbr of accidents per process batch
    nbr_accidents_in_batch = 500

    # init index keeping track of what we have processed
    process_index = weather_records_process_start_index

    while(process_index < accidents_count):

        # get accidents matched with weather data
        db_helper.match_accidents_with_weather_records(
            NBR_ACCIDENTS_IN_PROCESSED_BATCH=nbr_accidents_in_batch,
            WEATHER_STATION_MAX_DIST_FROM_ACCIDENT_IN_M=15000,
            WEATHER_DATA_MAX_TIME_DELTA_IN_S=7200,
            MATCH_WITH_N_NEAREST_WEATHER_STATIONS=5,
            OFFSET=process_index
        )

        # prompt
        logging.info(f'Accidents matched with weather records from index {process_index} to {process_index+nbr_accidents_in_batch}')

        # update
        process_index += nbr_accidents_in_batch

        # Wait
        time.sleep(5)



if __name__ == "__main__":

    # init
    threads = []

    # Thread 1
    p = threading.Thread(target=match_with_road_segments)
    threads.append(p)
    p.start()

    # Thread 2
    p = threading.Thread(target=match_with_weather_records)
    threads.append(p)
    p.start()

    # Start the Threads
    for index, thread in enumerate(threads):
        thread.join()
