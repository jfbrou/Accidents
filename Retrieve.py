# Retrieve the road traffic accidents data
if os.path.isfile(os.path.join(data, 'accidents.zip')) == False:
    url = 'http://donnees.ville.montreal.qc.ca/dataset/cd722e22-376b-4b89-9bc2-7c7ab317ef6b/resource/5a81f6c5-e3e7-4c0e-9ccf-a4ba2cab77ba/download/collisions_routieres.zip'
    urllib.urlretrieve(url, os.path.join(data, 'accidents.zip'))

# Retrieve the road segments data
if os.path.isfile(os.path.join(data, 'segments.zip')) == False:
    url = 'http://donnees.ville.montreal.qc.ca/dataset/984f7a68-ab34-4092-9204-4bdfcca767c5/resource/70c1f8c7-91a0-4553-b602-89c3edb959b5/download/geobase.zip'
    urllib.urlretrieve(url, os.path.join(data, 'segments.zip'))

# Retrieve the traffic lights data
if os.path.isfile(os.path.join(data, 'traffic.zip')) == False:
    url = 'http://donnees.ville.montreal.qc.ca/dataset/02ebdab9-cbf3-4f56-8c29-79fa0ed0ed2e/resource/b5153f25-37d0-4e5f-a367-2d02b3f1d826/download/intersectionsafeux.zip'
    urllib.urlretrieve(url, os.path.join(data, 'traffic.zip'))

# Retrieve the weather data
stations = [10761, 51157, 30165, 5415, 48374, 10873, 47888, 26856, 10843, 49608, 5237, 10762]
for station in stations:
    for year in range(2012, 2019+1):
        for month in range(1, 12+1):
            if os.path.isfile(os.path.join(data, 'weather_'+str(station)+'_'+str(year)+'_'+str(month)+'.csv')) == False:
                url = 'https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID='+str(station)+'&Year='+str(year)+'&Month='+str(month)+'&Day=14&timeframe=1&submit=Download+Data'
                urllib.urlretrieve(url, os.path.join(data, 'weather_'+str(station)+'_'+str(year)+'_'+str(month)+'.csv'))

# Append and save the weather data into a single data frame
if os.path.isfile(os.path.join(data, 'weather.csv')) == False:
    columns = ['Longitude (x)', 'Latitude (y)', 'Climate ID', 'Date/Time', 'Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)', 'Wind Dir (10s deg)', 'Wind Spd (km/h)', 'Visibility (km)', 'Stn Press (kPa)', 'Weather']
    weather = pd.DataFrame(columns=columns)
    for station in stations:
        for year in range(2012, 2019+1):
            for month in range(1, 12+1):
                weather = weather.append(pd.read_csv(os.path.join(data, 'weather_'+str(station)+'_'+str(year)+'_'+str(month)+'.csv'), usecols=columns), ignore_index=True)
    weather.to_csv(os.path.join(data, 'weather.csv'), index=False)
