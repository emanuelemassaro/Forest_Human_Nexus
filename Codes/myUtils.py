"""
Author: Emanuele Massaro
Date: 2024-10-03
Description: This script contains the main functions for the other scripts
Project: Forest Human Nexus paper
Version: 1.0
"""

import os
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd

import pylab as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm # Import the class specifically
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt

import rasterio
from rasterio.plot import show
from rasterio.mask import mask
from rasterio.transform import Affine

from osgeo import gdal

import cartopy.crs as ccrs
import cartopy.feature as cfeature
crsplot = ccrs.Mollweide()



MAINPATH = '/home/emanuele/Research/APES/FHN/'


REGION_TO_CONTINENT = {'Central Asia': 'Asia', 'Eastern Africa': 'Africa','Eastern Asia': 'Asia','Eastern Europe': 'Europe',
                        'Middle Africa': 'Africa', 'Northern Africa': 'Africa', 'Northern America': 'America',
                        'Northern Europe': 'Europe', 'Oceania': 'Oceania', 'South America': 'America', 'South-Eastern Asia': 'Asia',
                        'Southern Africa': 'Africa', 'Southern Asia': 'Asia', 'Southern Europe': 'Europe', 'Western Africa': 'Africa',
                        'Western Asia': 'Asia', 'Western Europe': 'Europe'}

CONTINENT_COLORS = {
    'Asia': '#E69F00',      # Bright Orange
    'Africa': '#0072B2',    # Dark Blue
    'Europe': '#009E73',    # Green
    'America': '#D55E00',   # Reddish-Orange
    'Oceania': '#F0E442'    # Yellow
}



#MAINPATH = '/home/emanuele/Research/APES/FHN/'
# Check if MAINPATH is empty or None
#if not MAINPATH:
#    # If MAINPATH is empty, write the mainpath
#    MAINPATH = input("Please enter the MAINPATH like 'home/FHN/': ")
print(f"MAINPATH is set to: {MAINPATH}")


def change_label_forest(arr):
    arr1 = arr.copy()

    # Set values between 40 and 45 (inclusive) to 2
    arr1[np.where((arr >= 40) & (arr <= 45))] = 2

    # Set values equal to 0 to 0
    arr1[np.where(arr == 0)] = 0

    # Set all other values to 1
    arr1[np.where((arr != 0) & (arr < 40) | (arr > 45))] = 1

    return arr1



def read_raster(file_path):
    """Read raster data from the given file path."""
    src = rasterio.open(file_path)
    data = src.read(1)
    return data, src

def save_output_raster(output_array, output_file_name, crs, transform, dtype, nodata):
    with rasterio.open(
        output_file_name, 
        'w', 
        driver='GTiff', 
        height=output_array.shape[0], 
        width=output_array.shape[1], 
        count=1, 
        dtype=dtype, 
        nodata=nodata, 
        crs=crs, 
        transform=transform,
        compression='LZW'
    ) as dst:
        dst.write(output_array, 1)

def set_font(ax, size):
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(size)
    return ax





def move_folders_with_pattern(source_folder, destination_folder, pattern):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Loop through all items in the source folder
    for item in os.listdir(source_folder):
        # Construct full path
        item_path = os.path.join(source_folder, item)

        # Check if it's a directory and matches the pattern
        if os.path.isdir(item_path) and pattern in item:
            # Construct destination path
            destination_path = os.path.join(destination_folder, item)

            # Move the directory
            shutil.move(item_path, destination_path)
            print(f"Moved {item_path} to {destination_path}")


def return_population_data_array(year, population_raster_pattern):
    population_raster_path = population_raster_pattern.format(year=year)
    src = rasterio.open(population_raster_path)
    out_image = src.read(1)
    out_image[out_image<0]=0
    return out_image

def return_distance_data_array(year, distance_to_forest_pattern):
    raster_path = distance_to_forest_pattern.format(year=year)
    src = rasterio.open(raster_path)
    out_image = src.read(1)
    out_image = -out_image
    out_image[out_image<0]=0
    return out_image


def return_forest_data_array(year, forest_raster_pattern):
    raster_path = forest_raster_pattern.format(year=year)
    src = rasterio.open(raster_path)
    forest_array = src.read(1)
    # Mask and process forest array
    mask_arr = (forest_array >= 40) & (forest_array <= 45)
    forest_array = np.where(mask_arr, 1.0, 0.0)
    return forest_array



def calculate_new_transform(original_transform, window_size):
    return Affine(
        original_transform[1] * window_size,
        original_transform[2],
        original_transform[0],
        original_transform[4],
        original_transform[5] * window_size,
        original_transform[3]
    )


def getCRSTransform(year, population_raster_pattern):
    population_raster_path = population_raster_pattern.format(year=year)
    population_raster = gdal.Open(population_raster_path)
    original_transform = population_raster.GetGeoTransform()
    original_crs = population_raster.GetProjection()
    return original_transform, original_crs


def initialize_output_dataset(data, WINDOWS):
    rows, cols = data.shape[0], data.shape[1]
    output_cols = int(cols / WINDOWS)
    output_rows = int(rows / WINDOWS)
    return output_cols, output_rows, np.full((output_rows, output_cols), np.nan, dtype=np.float32)


def plot_data_maps(ax, data, cmap, norm, src):
    """Plot the raster data with the provided axis, colormap, and normalization."""
    ax.set_extent([-18000000, 18000000, -9000000, 9000000], crs=crsplot)
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.add_feature(cfeature.LAND, facecolor='white')
    gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
    gl.ylocator = mticker.FixedLocator([-75, -45, 0, 45, 75])
    gl.top_labels = False
    gl.bottom_labels = False
    show(data, cmap=cmap, norm=norm, ax=ax, transform=src.transform)


def add_legend(ax, patches, title):
    """Add a legend to the plot."""
    legend = ax.legend(handles=patches, 
                       loc='center right', 
                       bbox_to_anchor=(1.3, 0.5), 
                       title=title, title_fontsize=14)
    legend.get_title().set_ha('center')
    



def return_df(PATHL, gdf1):
    # Ask the user for input
    window = 50
    YEARS = np.arange(1975, 2025, 5)
    cols = ['year', 'region', 'mean', 'median', 'stdev', 'all_vals', 'sum']
    df1 = pd.DataFrame(columns=cols)
    for year in YEARS:
        file_raster_path = PATHL.format(window=window, year=year)
        # Read raster data
        data, src = read_raster(file_raster_path)
        for index, region in gdf1.iterrows(): 
            dat, _ = mask(src, [region['geometry']], crop=True)
            dat_values = dat.flatten()
            dat_values = dat_values[~np.isnan(dat_values)]
            data = [year, gdf1['custom_reg'][index], np.nanmean(dat), np.nanmedian(dat), np.nanstd(dat), dat_values, np.nansum(dat)]
            df1 = pd.concat([df1, pd.DataFrame(columns=cols, data=[data])], ignore_index=True)
    return df1

def rel_change_df(df1, val):
    df_1975 = df1[df1['year'] == 1975]
    df_2020 = df1[df1['year'] == 2020]
    
    # Merge the data on the 'region' column to compare 1975 and 2020
    merged_df = pd.merge(df_1975[['region', val]], df_2020[['region', val]], 
                         on='region', suffixes=('_1975', '_2020'))
    col1 = val+'_2020'
    col2 = val+'_1975'
    # Calculate the relative change
    merged_df['relative_change'] = (merged_df[col1] - merged_df[col2]) / merged_df[col2] * 100
    #merged_df = merged_df.sort_values(by='relative_change')
    return merged_df