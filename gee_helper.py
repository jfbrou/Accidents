import os
from datetime import datetime
import ee
import requests

# Pretty print
import pprint
pp = pprint.PrettyPrinter(depth=4)

#ee.Authenticate()
ee.Initialize()

# Find the current working directory
path = os.getcwd()

# Grab path to data folder
if os.path.isdir(os.path.join(path, 'Data')) == False:
    raise Exception('Data directory does not exist, run retrieve script')
data_dir_path = os.path.join(path, 'Data')


def humansize(nbytes):
    """
        Function to get sizes in Human readable format
    """

    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']

    i = 0
    while nbytes >= 1024 and i < len(suffixes)-1:
        nbytes /= 1024.
        i += 1
    f = ('%.2f' % nbytes).rstrip('0').rstrip('.')
    return '%s %s' % (f, suffixes[i])


def envelope_polygon_to_corners(geometry):
    """
        Takes an envelope polygon as input and returns the corners
    """

    # cast as str
    geometry = str(geometry)

    # parse
    geometry = geometry.replace('POLYGON((', '')
    geometry = geometry.replace('))', '')
    geometry = geometry.strip()

    # split points
    points = geometry.split(',')
    if(len(points) != 5):
        raise Exception('Envelope Polygon provided as input is invalid')

    # go through
    clean = []
    for pt in points:

        # split
        pt = pt.strip()
        pts = pt.split(' ')

        if(len(pts) == 2):
            clean.append([float(pts[0]), float(pts[1])])
        else:
            print(f'ERROR: Not a tuple ({pt})')

    # grab corners
    MIN_X = clean[0][0]
    MIN_Y = clean[0][1]

    MAX_X = clean[2][0]
    MAX_Y = clean[2][1]

    return [MIN_X, MIN_Y, MAX_X, MAX_Y]



def get_imagery(
        ROI_GEOMETRY_RECT,
        DATE_LOWER_BOUND='2019-07-21',
        DATE_UPPER_BOUND='2019-08-01',
        IMAGE_COLLECTION='LANDSAT/LC08/C01/T1',
        INTEREST_BANDS=['B5', 'B4', 'B3', 'B2'],
        MAX_CLOUD_COVER=10
    ):

    # Print roi area in square kilometers.
    ROIarea = ROI_GEOMETRY_RECT.area().divide(1000 * 1000).getInfo()
    print(f'ROI area : {ROIarea} km^2')

    #Public Image Collections
    results = ee.ImageCollection(IMAGE_COLLECTION).filterDate(DATE_LOWER_BOUND, DATE_UPPER_BOUND).filterBounds(ROI_GEOMETRY_RECT).filterMetadata('CLOUD_COVER','less_than', MAX_CLOUD_COVER)

    # Get collection size
    print('Total number of assets with filters: '+str(results.size().getInfo()))
    print('\n'+'Total size of collection : '+str(humansize(results.reduceColumns(ee.Reducer.sum(), ['system:asset_size']).getInfo()['sum'])))

    # Create a list with all the images
    collectionList = results.toList(results.size())
    collectionSize = collectionList.size().getInfo()

    # Parse
    for i in range(0, collectionSize):

        # get data
        infoDict = collectionList.get(i).getInfo()

        # id
        print(f"--- Index: {i} ---")
        print(f"\nid:\t\t\t\t{infoDict['id']}")

        # date acquired
        print(f"\nDATE_ACQUIRED:\t\t\t{infoDict['properties']['DATE_ACQUIRED']}")

        # times
        time_start = str(datetime.fromtimestamp(infoDict['properties']['system:time_start']/1000.0))
        time_end = str(datetime.fromtimestamp(infoDict['properties']['system:time_end']/1000.0))

        print(f"Time Start: \t\t\t{time_start}")
        print(f"Time End: \t\t\t{time_end}")

        # other info
        print(f"Sun Elevation: \t\t\t{infoDict['properties']['SUN_ELEVATION']}")

        # bands
        print('\n--- Bands ---')
        for band in infoDict['bands']:
            print(f"id: {band['id']}, crs: {band['crs']}, max: {band['data_type']['max']}")

        print('\n\n\n\n')

        # STOP AFTER FIRST
        break

    # Selected image index
    desiredIndex = 0

    # Get an image
    sample_image = ee.Image(collectionList.get(desiredIndex)).select(INTEREST_BANDS)

    # print all info for this asset
    pp.pprint(sample_image.getInfo())

    # clip
    clipped_img = sample_image.clip(ROI_GEOMETRY_RECT)

    # get url
    asset_url = clipped_img.getDownloadURL()

    return asset_url


def download_asset(url, filename):

    # get request
    r = requests.get(url, allow_redirects=True)

    # init filename
    outname = filename

    # check file type
    filetype = r.headers.get('content-type')
    if('zip' in filetype):
        outname = outname + '.zip'
    elif('png' in filetype):
        outname = outname + '.png'
    elif('tif' in filetype):
        outname = outname + '.tif'
    else:
        print(filetype)

    # write to file
    outpath = os.path.join(data_dir_path, outname)
    open(outpath, 'wb').write(r.content)

    print(f'Download done {outpath}')


# Test
envelope_polygon = 'POLYGON((-73.7068029846723 45.5265622667585,-73.7068071182513 45.5276903170224,-73.705155796456 45.5276932946443,-73.7051516958824 45.5265652442639,-73.7068029846723 45.5265622667585))'
corners = envelope_polygon_to_corners(envelope_polygon)

# Production of rectangle limiting ROI - it's a ee.geometry defined from coordinates and staying in the GEE.
ROIgeometry = ee.Geometry.Rectangle(coords=corners, proj='EPSG:4326')

# grab url of imagery
url = get_imagery(ROIgeometry, INTEREST_BANDS=['B3', 'B2'])

# download
#download_asset(url, 'mtl-feature')