# Retrieve the road traffic accidents data
if os.path.isfile(os.path.join(data, 'accidents.csv')) == False:
    urlaccidents = 'http://donnees.ville.montreal.qc.ca/dataset/cd722e22-376b-4b89-9bc2-7c7ab317ef6b/resource/05deae93-d9fc-4acb-9779-e0942b5e962f/download/collisions_routieres.csv'
    urllib.urlretrieve(urlaccidents, os.path.join(data, 'accidents.csv'))

# Retrieve the road segments data
if os.path.isfile(os.path.join(data, 'segments.zip')) == False:
    urlsegments = 'http://donnees.ville.montreal.qc.ca/dataset/984f7a68-ab34-4092-9204-4bdfcca767c5/resource/70c1f8c7-91a0-4553-b602-89c3edb959b5/download/geobase.zip'
    urllib.urlretrieve(urlsegments, os.path.join(data, 'segments.zip'))
