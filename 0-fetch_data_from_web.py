################################################################################
#                                                                              #
# This script fetches all the source data from different web endpoints         #
# and writes everything as csvs in the Data/ folder                            #
#                                                                              #
################################################################################


import os
import urllib.request as urllib
import pandas as pd

# Logging
import logging
logging.basicConfig(level=logging.INFO)

# Find the current working directory
path = os.getcwd()

# Create a folder that contains all data files
if os.path.isdir(os.path.join(path, 'Data')) == False:
    os.mkdir('Data')
data_dir_path = os.path.join(path, 'Data')

# Retrieve the road traffic accidents data
out_path_accidents = os.path.join(data_dir_path, 'accidents.csv')
if os.path.isfile(out_path_accidents) == False:
    urlaccidents = 'http://donnees.ville.montreal.qc.ca/dataset/cd722e22-376b-4b89-9bc2-7c7ab317ef6b/resource/05deae93-d9fc-4acb-9779-e0942b5e962f/download/collisions_routieres.csv'
    urllib.urlretrieve(urlaccidents, out_path_accidents)
    logging.info(f'Finished downloading {out_path_accidents}')
else:
    logging.info(f'Already downloaded {out_path_accidents}')

# Retrieve the road segments data
out_path_segments = os.path.join(data_dir_path, 'segments.zip')
if os.path.isfile(out_path_segments) == False:
    urlsegments = 'http://donnees.ville.montreal.qc.ca/dataset/984f7a68-ab34-4092-9204-4bdfcca767c5/resource/70c1f8c7-91a0-4553-b602-89c3edb959b5/download/geobase.zip'
    urllib.urlretrieve(urlsegments, out_path_segments)
    logging.info(f'Finished downloading {out_path_segments}')
else:
    logging.info(f'Already downloaded {out_path_segments}')


# Build weather data dataframe
weather_df = []
stations = [10761, 51157, 30165, 5415, 48374, 10873, 47888, 26856, 10843, 49608, 5237, 10762]
for station in stations:
    for year in range(2012, 2019+1):
        for month in range(1, 12+1):
            weather_df.append({
                'station_id': station,
                'year': year,
                'month': month,
                'out_path': os.path.join(data_dir_path, 'weather_'+str(station)+'_'+str(year)+'_'+str(month)+'.csv'),
                'url': 'https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID='+str(station)+'&Year='+str(year)+'&Month='+str(month)+'&Day=14&timeframe=1&submit=Download+Data'
            })


# Retrieve the weather data
for weather_datum in weather_df:

    out_path = weather_datum['out_path']
    url = weather_datum['url']

    if os.path.isfile(out_path) == False:
        urllib.urlretrieve(url, out_path)
        logging.info(f'Finished downloading {out_path}')
    else:
        logging.info(f'Already downloaded {out_path}')


# Append and save the weather data into a single data frame
out_path_weather_df = os.path.join(data_dir_path, 'weather.csv')
if os.path.isfile(out_path_weather_df) == False:
    columns = ['Longitude (x)', 'Latitude (y)', 'Climate ID', 'Date/Time', 'Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)', 'Wind Dir (10s deg)', 'Wind Spd (km/h)', 'Visibility (km)', 'Stn Press (kPa)', 'Weather']
    weather = pd.DataFrame(columns=columns)
    for weather_datum in weather_df:
        out_path = weather_datum['out_path']
        url = weather_datum['url']
        weather = weather.append(pd.read_csv(out_path, usecols=columns), ignore_index=True)

    weather.to_csv(out_path_weather_df, index=False)
    logging.info(f'Finished building {out_path_weather_df}')
else:
    logging.info(f'Already built {out_path_weather_df}')
