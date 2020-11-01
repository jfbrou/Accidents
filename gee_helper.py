import os
from datetime import datetime

from io import BytesIO
import requests
import zipfile

import ee

# image
from PIL import Image
import numpy as np

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


def bbox_to_corners(geometry):
    """
        Takes a postgis Box2D as input and returns the corners
    """

    # cast as str
    geometry = str(geometry)

    # parse
    geometry = geometry.replace('BOX(', '')
    geometry = geometry.replace(')', '')
    geometry = geometry.strip()

    # split points
    points = geometry.split(',')
    if(len(points) != 2):
        raise Exception('BBox provided as input is invalid')

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

    MAX_X = clean[1][0]
    MAX_Y = clean[1][1]

    return [MIN_X, MIN_Y, MAX_X, MAX_Y]


def get_imagery(
        BBOX_CORNERS,
        DATE_LOWER_BOUND='2015-02-01',
        DATE_UPPER_BOUND='2020-09-01',
        IMAGE_COLLECTION='COPERNICUS/S2_SR',
        INTEREST_BANDS=['B4', 'B3', 'B2'],
        MAX_CLOUD_COVER=20,
        MAX_VISUALIZE_INTENSITY=5000
    ):

    # check inputs
    if(len(INTEREST_BANDS) > 3):
        raise Exception('No more than 3 bands')

    # Production of rectangle limiting ROI
    ROI_GEOMETRY_RECT = ee.Geometry.Rectangle(coords=BBOX_CORNERS, proj='EPSG:4326')

    # Print roi area in square kilometers.
    ROIarea = ROI_GEOMETRY_RECT.area().divide(1000 * 1000).getInfo()
    print(f'ROI area : {ROIarea} km^2')

    #Public Image Collections
    results = ee.ImageCollection(IMAGE_COLLECTION).filterDate(DATE_LOWER_BOUND, DATE_UPPER_BOUND).filterBounds(ROI_GEOMETRY_RECT).filterMetadata('CLOUDY_PIXEL_PERCENTAGE','less_than', MAX_CLOUD_COVER)

    # Get collection size
    assets_count = results.size().getInfo()
    assets_size = humansize(results.reduceColumns(ee.Reducer.sum(), ['system:asset_size']).getInfo()['sum'])
    print(f'Total number of assets with filters: {assets_count}\n')
    print(f'Total size of collection : {assets_size}')

    # check if we found assets
    if(assets_count == 0):
        return None

    # Create a list with all the images
    collectionList = results.toList(results.size())
    collectionSize = collectionList.size().getInfo()

    # Parse
    for i in range(0, collectionSize):

        # get data
        infoDict = collectionList.get(i).getInfo()

        # print index
        print(f"--- Index: {i} ---")
        print(f"\nid:\t\t\t\t{infoDict['id']}")

        # print info
        pp.pprint(infoDict)

        print('\n\n\n\n')

        # STOP AFTER FIRST
        break

    # Selected image index
    desiredIndex = 0

    # Get an image
    sample_image = ee.Image(collectionList.get(desiredIndex)).select(INTEREST_BANDS)

    # visualize
    img_visualized = sample_image.visualize(
        min=0,
        max=MAX_VISUALIZE_INTENSITY
    )

    # clip
    clipped_img = img_visualized.clipToBoundsAndScale(
        geometry=ROI_GEOMETRY_RECT
    )

    # get url
    asset_url = clipped_img.getDownloadURL()

    return asset_url


def download_asset(url, asset_id):

    # get request
    r = requests.get(url, allow_redirects=True)

    # check file type
    filetype = r.headers.get('content-type')

    # if zip
    if('zip' in filetype):

        # Create a folder that contains all data files
        dir_path = os.path.join(data_dir_path, asset_id)
        if os.path.isdir(dir_path) == False:
            os.mkdir(dir_path)

        # read bytes
        filebytes = BytesIO(r.content)

        # unzip
        filenames = []
        with zipfile.ZipFile(filebytes, 'r') as zip_ref:

            # set files
            filenames = zip_ref.namelist()

            # extract
            zip_ref.extractall(dir_path)

        # compose
        if(len(filenames) > 1):

            # init bands
            bands = []

            for filename in filenames:

                # load grayscale band
                band_img = Image.open(os.path.join(dir_path, filename))

                # convert to numpy array
                band = np.asarray(band_img)

                # append
                bands.append(band)

            # stack bands
            canvas = np.dstack(bands)

            # save image
            img = Image.fromarray(canvas)
            img.save(os.path.join(dir_path, 'composed.png'))

    else:

        if('png' in filetype):
            asset_id = asset_id + '.png'
        elif('tif' in filetype):
            asset_id = asset_id + '.tif'
        else:
            print(filetype)

        # write to file
        outpath = os.path.join(data_dir_path, asset_id)
        open(outpath, 'wb').write(r.content)

    print(f'Download done {url}')
