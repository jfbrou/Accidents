# Import libraries
import os
import datetime

import pandas as pd
import numpy as np
import geopandas as gpd

#from pvlib import solarposition

from config import username, password, hostname, port, db_name
from sqlalchemy import create_engine, types
from geoalchemy2.types import Geometry
import psycopg2

# Set up database connection engine
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


################################################################################
#                                                                              #
# This section of the script preprocesses the road segments data.              #
#                                                                              #
################################################################################

# Load the data
segments = gpd.read_file(r'zip://'+os.path.join(data_dir_path, 'segments.zip'))

# Keep relevant columns
segments = segments.loc[:, ['ID_TRC', 'CLASSE', 'SENS_CIR', 'geometry']]

# Rename columns
segments = segments.rename(columns={'ID_TRC':'segment_id', 'CLASSE':'class', 'SENS_CIR':'direction'})

# Redefine the types of each column
segments = segments.astype({'segment_id':'int64', 'class':'int64', 'direction':'int64'})

# Reset the data frame's index
segments = segments.reset_index(drop=True)


# add to db
try:

    segments.to_postgis(
        con=engine,
        name='road_segments',
        if_exists='replace',
        dtype={
            'segment_id': types.Integer(),
            'class': types.Integer(),
            'direction': types.Integer(),
            'geometry': Geometry(geometry_type='LINESTRING', srid=32188)
        }
    )

    # log
    logging.info(f'Segments loaded')

except ValueError as err:
    print(err)
except:
    logging.warning('Did not load segments into database')



################################################################################
#                                                                              #
# This section of the script preprocesses the road traffic accidents data.     #
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

# Reset the data frame's index
accidents = accidents.reset_index(drop=True)


try:
    # add to database
    accidents.to_postgis(
        con=engine,
        name='accidents',
        if_exists='replace',
        dtype={
            'accident_id': types.Text(),
            'datetime': types.DateTime(),
            'geometry': Geometry(geometry_type='POINT', srid=32188)
        }
    )

    # log
    logging.info(f'Accidents loaded')


except ValueError as err:
    print(err)

except psycopg2.errors as err:
    print(err)

except:
    logging.warning('Did not load accidents into database')
