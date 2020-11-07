import os

# Lib for tabular data processing
import numpy as np
import pandas as pd
import geopandas as gpd

# Lib to import our accidents data
import db_helper

# Lib to work with .tif
import imagery_helper

# Find the current working directory
path = os.getcwd()

# Grab path to data folder
if os.path.isdir(os.path.join(path, 'Data')) == False:
    raise Exception('Data directory does not exist, run retrieve script')
data_dir_path = os.path.join(path, 'Data')

################################################################################
#                                                                              #
#                  Loading a sample the processed accidents                    #
#                                                                              #
################################################################################

# Load the data
df = pd.read_csv(os.path.join(data_dir_path, 'data.csv'))

# remove nan
df = df.dropna()

# Keep certain columns
df = df.loc[:, ['accident_id',  'accident_datetime', 'road_segment_id', 'road_segment_bbox']]


################################################################################
#                                                                              #
#                       Loading our imagery of Montreal                        #
#                                                                              #
################################################################################

# Path to our imagery of montreal
image_file = "/home/jean-romain/Geospatial/30cm_imagery/montreal.tif"

# load
satdat = imagery_helper.load(image_file)

# display info
imagery_helper.info(satdat)

# show
#imagery_helper.show(satdat)


################################################################################
#                                                                              #
#                     Go through accidents and crop image                      #
#                                                                              #
################################################################################

# Go through rows
for index, row in df.iterrows():

    # get values
    accident_id = row['accident_id']
    road_segment_id = row['road_segment_id']
    road_segment_bbox = row['road_segment_bbox']
    accident_datetime = row['accident_datetime']

    # get area of interest
    print(road_segment_bbox)

    # crop
    #imagery_helper.crop(image_file, 'cropped.tif', NONE)

    break


# # compress
# imagery_helper.compress(image_file, 'compressed.tif')


# # reproject
# imagery_helper.reproject(image_file, 'reprojected.tif')
