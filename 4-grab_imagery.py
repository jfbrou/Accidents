
# Import libraries
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import gee helper
import gee_helper

# Logging
import logging
logging.basicConfig(level=logging.INFO)

# Find the current working directory
path = os.getcwd()

# Grab path to data folder
if os.path.isdir(os.path.join(path, 'Data')) == False:
    raise Exception('Data directory does not exist, run retrieve script')
data_dir_path = os.path.join(path, 'Data')

# Grab path to figures folder
if os.path.isdir(os.path.join(path, 'Figures')) == False:
    os.mkdir('Figures')
figures_dir_path = os.path.join(path, 'Figures')


################################################################################
#                                                                              #
#                        Loading the data from the DB                          #
#                                                                              #
################################################################################


# Load the data
df = pd.read_csv(os.path.join(data_dir_path, 'data.csv'))

# remove nan
df = df.dropna()

# Keep certain columns
df = df.loc[:, ['accident_id',  'accident_datetime', 'road_segment_id', 'road_segment_bbox']]

# Go through rows
for index, row in df.iterrows():

    # get values
    accident_id = row['accident_id']
    road_segment_id = row['road_segment_id']
    road_segment_bbox = row['road_segment_bbox']
    accident_datetime = row['accident_datetime']

    # grab corners of bbox
    corners = gee_helper.bbox_to_corners(road_segment_bbox)
    print(corners)

    # get current datetime
    now = datetime.now()
    current_year = now.year

    # get datetime
    dateformat = '%Y-%m-%d %H:%M:%S'
    datetime_obj = datetime.strptime(accident_datetime, dateformat)
    adjust_year = datetime_obj.year
    if(adjust_year < 2017):
        adjust_year = 2017
        datetime_obj = datetime_obj.replace(year=adjust_year)

    # if cant find assets for date looks at the next year for the ~same month
    url = None
    while(url is None or adjust_year > current_year):

        # set time window
        date_target = datetime_obj.strftime('%Y-%m-%d')

        # prompt
        print(f'Fetching {accident_id} - {date_target}')

        # grab url of imagery
        url = gee_helper.get_imagery(corners, FILTER_DATE=date_target, FILTER_DATE_RADIUS_DAYS=30)

        # increment year
        adjust_year += 1
        datetime_obj = datetime_obj.replace(year=adjust_year)

    # outpath
    outpath = str(accident_id) + "_" + str(road_segment_id)

    # download
    if url is not None:
        gee_helper.download_asset(url, outpath)

    # TEMP
    break
