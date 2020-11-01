
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

# grab corners of bbox
corners = [-73.812302, 45.402782, -73.551460, 45.673703]

# grab url of imagery
gee_helper.get_imagery(
    corners,
    FILTER_DATE='2020-01-01',
    FILTER_DATE_RADIUS_DAYS=60,
    MAX_VISUALIZE_INTENSITY=15000,
    EXPORT_TO_DRIVE=True
)