
import numpy as np
import pandas as pd
import geopandas as gpd
from osgeo import gdal
import ast
import os


# Rasterio
import rasterio
from rasterio.plot import show
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.mask import mask

# Plotting 
import matplotlib.pylab as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm # Import the class specifically
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import scipy.stats as stats

import cartopy.crs as ccrs
import cartopy.feature as cfeature
crsplot = ccrs.Mollweide()

PATHRESEARCH = '/home/emanuele/Research/'
PATHAPES = os.path.join(PATHRESEARCH, 'APES')


PATHD = '/home/emanuele/Research/APES/Data/Landuse/proximity_{window}_{year}.tif'           ## Proximity 
PATHP = '/home/emanuele/Research/APES/Data/Landuse/population_{window}_{year}_v5.tif'       ## Population
PATHF = '/home/emanuele/Research/APES/Data/Landuse/forest_{window}_{year}.tif'              ## Forest
PATHA = '/home/emanuele/Research/APES/Data/Landuse/fap_{window}_{year}.tif'                 ## FAP
PATHN = '/home/emanuele/Research/APES/Data/Landuse/FHN_{window}_{year}_final.tif'                 ## FHN
PATHFPP = '/home/emanuele/Research/APES/Data/Landuse/FPP_{window}_{year}.tif'                 ## FPP
PATHS = [PATHD, PATHP, PATHF, PATHA, PATHN, PATHFPP]
NAMES = ['Proximity', 'Population', 'Forest', 'FAP', 'FHN', 'FPP']

## Define costum regions ####################################################################################################################
fout = '/home/emanuele/Research/Data/Shapefiles/world-custom_regions_r.shp'
gdf1 = gpd.read_file(fout)


def setFont(ax, size):
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(size)
    return ax



def read_raster(file_path):
    """Read raster data from the given file path."""
    src = rasterio.open(file_path)
    data = src.read(1)
    return data, src
def returnDF():
    # Ask the user for input
    j = int(input("Enter the index value for j: 0: Proximiy, 1: Population, 2: Forest, 3: FAP, 4: FHN, 5: FPP ---->  "))
    PATHL = PATHS[j]
    window = 50
    YEARS = np.arange(1975, 2025, 5)
    cols = ['year', 'region', 'mean', 'median', 'stdev', 'all_vals', 'sum']
    df1 = pd.DataFrame(columns=cols)
    distr = []
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

def returnRelChange(df1, val):
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

def returnTrend(df1, val):
    df_1975 = df1[df1['year'] == 1975]
    df_2020 = df1[df1['year'] == 2020]
    
    # Merge the data on the 'region' column to compare 1975 and 2020
    merged_df = pd.merge(df_1975[['region', val]], df_2020[['region', val]], 
                         on='region', suffixes=('_1975', '_2020'))
    col1 = val+'_2020'
    col2 = val+'_1975'
    # Calculate the relative change
    merged_df['relative_change'] = (merged_df[col1] - merged_df[col2]) / (2020-1975)
    #merged_df = merged_df.sort_values(by='relative_change')
    return merged_df

def set_font(ax, fontsize):
    """Set font size for the axis."""
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    return ax

def plotBar(merged_df, title, ylabel, ascending):
    merged_df = merged_df.sort_values(by='relative_change', ascending=ascending)
    # Plotting the bar plot
    f, ax = plt.subplots(figsize=(10, 4.5))
    ax = set_font(ax, 14)
    plt.bar(merged_df['region'], merged_df['relative_change'], color='skyblue')
    
    # Adding labels and title
    plt.xlabel(' ')
    plt.ylabel(ylabel, size=15)
    plt.title(title, size=16)
    plt.xticks(rotation=90, ha='right')
    plt.grid(axis='y')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return f
