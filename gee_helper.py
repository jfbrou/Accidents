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


def return_ee_geometry(geometry):
    """
        Takes a single feature as input and returns the google earth geometry
    """

    # cast as str
    geometry = str(geometry)

    # parse
    geometry = geometry.split('(')[-1]
    geometry = geometry.replace(')', '')
    geometry = geometry.strip()

    # split points
    points = geometry.split(',')
    print(f'Number of points: {len(points)}')

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

    return clean


def get_imagery(
        ROI_GEOMETRY,
        DATE_LOWER_BOUND='2019-07-21',
        DATE_UPPER_BOUND='2019-08-01',
        IMAGE_COLLECTION='LANDSAT/LC08/C01/T1',
        INTEREST_BANDS=['B5', 'B4', 'B3', 'B2'],
        MAX_CLOUD_COVER=10
    ):

    #Public Image Collections
    results = ee.ImageCollection(IMAGE_COLLECTION).filterDate(DATE_LOWER_BOUND, DATE_UPPER_BOUND).filterBounds(ROI_GEOMETRY).filterMetadata('CLOUD_COVER','less_than', MAX_CLOUD_COVER)

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
    clipped_img = sample_image.clip(ROI_GEOMETRY)

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
feature = 'LINESTRING(288690.550115 5043101.721475,288685.92721 5043096.320675,288654.88053 5043060.83725,288653.173125 5043058.62705,288651.993625 5043056.65685,288651.48224 5043055.276725,288650.93785 5043053.0165,288650.896585 5043050.226225,288651.193525 5043048.50605,288651.57296 5043047.145925,288652.636995 5043044.8057,288653.88247 5043043.045525,288709.49264 5042991.570575,288713.31989 5042988.31025,288714.59837 5042987.570175,288717.26257 5042986.610075,288719.38427 5042986.3438,288722.49199 5042986.390075,288723.902455 5042986.7001,288726.632665 5042987.8602,288727.74619 5042988.570275,288729.931975 5042990.41045,288765.391525 5043023.293625,288769.90565 5043028.6914)'
feature = return_ee_geometry(feature)

# Production of rectangle limiting ROI - it's a ee.geometry defined from coordinates and staying in the GEE.
#ROIgeometry = ee.Geometry.Polygon(feature)
ROIgeometry = ee.Geometry.LineString(feature)

# grab url of imagery
url = get_imagery(ROIgeometry, INTEREST_BANDS=['B3', 'B2'])

# download
download_asset(url, 'mtl-feature')
