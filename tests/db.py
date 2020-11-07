# libs
import pandas as pd
import geopandas as gpd

# Append parent folder
import sys
sys.path.insert(0,'..')

# import a method from db helper
from db_helper import get_accidents

# test
accidents = get_accidents(LIMIT=10)
if type(accidents) != gpd.geodataframe.GeoDataFrame:
    raise Exception('Error')

print('Test passed')
