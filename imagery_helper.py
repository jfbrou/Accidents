import os
from matplotlib import pyplot as plt
from humanize import naturalsize as sz

# import rasterio's tools
import rasterio
from rasterio.plot import show as rasterio_show
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform
from rasterio.warp import reproject as rasterio_reproject

# Pretty print
import pprint
pp = pprint.PrettyPrinter(depth=4)


def load(src_path):
    """
        Loads imagery as a rasterio object
    """
    satdat = rasterio.open(src_path)
    return satdat


def show(satdat):
    """
        Display satdat as matplotlib figure
    """

    # check input
    if(type(satdat) != rasterio.io.DatasetReader):
        raise Exception('Wrong Format')

    rasterio_show(satdat)


def info(satdat):
    """
        Display General Info about the satdat
    """

    # check input
    if(type(satdat) != rasterio.io.DatasetReader):
        raise Exception('Wrong Format')

    # dataset name
    print(f'Dataset Name : {satdat.name}\n')

    # number of bands in this dataset
    print(f'Number of Bands : {satdat.count}\n')

    # The dataset reports a band count.
    print(f'Number of Bands according to dataset : {satdat.count}\n')

    # And provides a sequence of band indexes.  These are one indexing, not zero indexing like Numpy arrays.
    print(f'Bands indexes : {satdat.indexes}\n')

    # Minimum bounding box in projected units
    print(f'Min Bounding Box : {satdat.bounds}\n')

    # Get dimensions, in map units (using the example GeoTIFF, that's meters)
    width_in_projected_units = satdat.bounds.right - satdat.bounds.left
    height_in_projected_units = satdat.bounds.top - satdat.bounds.bottom
    print(f"Width: {width_in_projected_units}, Height: {height_in_projected_units}\n")

    # Number of rows and columns.
    print(f"Rows: {satdat.height}, Columns: {satdat.width}\n")

    # This dataset's projection uses meters as distance units.  What are the dimensions of a single pixel in meters?
    xres = (satdat.bounds.right - satdat.bounds.left) / satdat.width
    yres = (satdat.bounds.top - satdat.bounds.bottom) / satdat.height
    print(f'Width of pixel (in m) : {xres}')
    print(f'Height of pixel (in m) : {yres}')
    print(f"Are the pixels square: {xres == yres}\n")

    # Get coordinate reference system
    print(f'Coordinates System : {satdat.crs}\n')

    # Convert pixel coordinates to world coordinates.
    # Upper left pixel
    row_min = 0
    col_min = 0

    # Lower right pixel.  Rows and columns are zero indexing.
    row_max = satdat.height - 1
    col_max = satdat.width - 1

    # Transform coordinates with the dataset's affine transformation.
    topleft = satdat.transform * (row_min, col_min)
    botright = satdat.transform * (row_max, col_max)

    print(f"Top left corner coordinates: {topleft}")
    print(f"Bottom right corner coordinates: {botright}\n")

    # All of the metadata required to create an image of the same dimensions, datatype, format, etc. is stored in
    # the dataset's profile:
    pp.pprint(satdat.profile)
    print('\n')


def compress(src_path, out_path, compression_type='JPEG'):
    """
        Compress imagery to reduce filesize

        Raster datasets use compression to reduce filesize. There are a number of compression methods,
        all of which fall into two categories: lossy and lossless. Lossless compression methods retain
        the original values in each pixel of the raster, while lossy methods result in some values
        being removed. Because of this, lossy compression is generally not well-suited for analytic
        purposes, but can be very useful for reducing storage size of visual imagery.

        By creating a lossy-compressed copy of a visual asset, we can significantly reduce the
        dataset's filesize. In this example, we will create a copy using the "JPEG" lossy compression method
    """

    # get initial size in bytes
    init_size = os.path.getsize(src_path)

    # load file
    satdat = load(src_path)

    # read all bands from source dataset into a single 3-dimensional ndarray
    data = satdat.read()

    # write new file using profile metadata from original dataset and specifying compression type
    profile = satdat.profile
    profile['compress'] = compression_type

    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(data)

    # returns size in bytes
    final_size = os.path.getsize(out_path)

    # compute diff
    ratio = round(10000.0*final_size/float(init_size))/100.0

    # output a human-friendly size
    print(f'(Initial size, Final Size, Ratio) : ({sz(init_size)}, {sz(final_size)}, {ratio}%)\n')


def crop(src_path, out_path, aoi_geojson):
    """
        Crop imagery using a GeoJson containing the area of interest
    """

    {
        "type": "Polygon",
        "coordinates": [
            [
                [ 546021.252692307694815, 4185843.204663460608572 ],
                [ 556143.537403846159577, 4186685.32485576858744 ],
                [ 554543.509038461605087, 4180436.793028845451772 ],
                [ 546408.627980769262649, 4179763.09687499934807 ],
                [ 546021.252692307694815, 4185843.204663460608572 ]
            ]
        ]
    }

    # apply mask with crop=True to crop the resulting raster to the AOI's bounding box
    with rasterio.open(src_path) as satdata:
        clipped, transform = mask(satdata, aoi, crop=True)

    # Using a copy of the metadata from our original raster dataset, we can write a new geoTIFF
    # containing the new, clipped raster data:
    meta = satdata.meta.copy()

    # update metadata with new, clipped mosaic's boundaries
    meta.update(
        {
            "transform": transform,
            "height":clipped.shape[1],
            "width":clipped.shape[2]
        }
    )

    # write the clipped-and-cropped dataset to a new GeoTIFF
    with rasterio.open(out_path, 'w', **meta) as dst:
        dst.write(clipped)


def reproject(src_path, out_path, target_crs='EPSG:4326'):
    """
        Reprojects the imagery to a new coordinate system

        In order to translate pixel coordinates in a raster dataset into coordinates that use a
        spatial reference system, an **affine transformation** must be applied to the dataset.
        This **transform** is a matrix used to translate rows and columns of pixels into (x,y)
        spatial coordinate pairs. Every spatially referenced raster dataset has an affine transform
        that describes its pixel-to-map-coordinate transformation.

        In order to reproject a raster dataset from one coordinate reference system to another,
        rasterio uses the **transform** of the dataset: this can be calculated automatically using
        rasterio's `calculate_default_transform` method:

        target CRS: rasterio will accept any CRS that can be defined using WKT
    """

    # load satdata
    satdata = load(src_path)

    # calculate a transform and new dimensions using our dataset's current CRS and dimensions
    transform, width, height = calculate_default_transform(satdata.crs,
                                                        target_crs,
                                                        satdata.width,
                                                        satdata.height,
                                                        *satdata.bounds)

    # Using a copy of the metadata from the clipped raster dataset and the transform we defined above,
    # we can write a new geoTIFF containing the reprojected and clipped raster data:
    metadata = satdata.meta.copy()

    # Change the CRS, transform, and dimensions in metadata to match our desired output dataset
    metadata.update({'crs':target_crs,
                    'transform':transform,
                    'width':width,
                    'height':height})

    # apply the transform & metadata to perform the reprojection
    with rasterio.open('clipped_4326.tif', 'w', **metadata) as reprojected:
        for band in range(1, satdata.count + 1):
            rasterio_reproject(
                source=rasterio.band(satdata, band),
                destination=rasterio.band(reprojected, band),
                src_transform=satdata.transform,
                src_crs=satdata.crs,
                dst_transform=transform,
                dst_crs=target_crs
            )
