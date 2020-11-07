import os
from matplotlib import pyplot as plt
from humanize import naturalsize as sz

# import rasterio's tools
import rasterio
from rasterio.plot import show
from rasterio.mask import mask

# Pretty print
import pprint
pp = pprint.PrettyPrinter(depth=4)

# This notebook explores a single 4 band (blue, green, red, NIR) PlanetScope scene in a UTM projection.
image_file = "/home/jean-romain/Geospatial/30cm_imagery/20190818T175419Z_615_POM1_ST2_P.tif"
image_file = "/home/jean-romain/Geospatial/30cm_imagery/montreal.tif"

# open
satdat = rasterio.open(image_file)

# satdat is our open dataset object
print('### General Info ###')
print(satdat)
print('\n')

# show
show(satdat)

# --- let's look at some basic information about this geoTIFF: ---

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


"""
File Compression
Raster datasets use compression to reduce filesize. There are a number of compression methods, all of which fall into two categories: lossy and lossless. Lossless compression methods retain the original values in each pixel of the raster, while lossy methods result in some values being removed. Because of this, lossy compression is generally not well-suited for analytic purposes, but can be very useful for reducing storage size of visual imagery.

All Planet data products are available as GeoTIFFs using lossless LZW compression. By creating a lossy-compressed copy of a visual asset, we can significantly reduce the dataset's filesize. In this example, we will create a copy using the "JPEG" lossy compression method:
"""

# returns size in bytes
size = os.path.getsize(image_file)

# output a human-friendly size
print(f'Raw image size : {sz(size)}')

# read all bands from source dataset into a single 3-dimensional ndarray
data = satdat.read()

# write new file using profile metadata from original dataset
# and specifying JPEG compression
profile = satdat.profile
profile['compress'] = 'JPEG'

with rasterio.open('compressed.tif', 'w', **profile) as dst:
    dst.write(data)

# returns size in bytes
size = os.path.getsize('compressed.tif')

# output a human-friendly size
print(f'Compressed image size : {sz(size)}\n')

