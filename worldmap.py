# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 12:05:49 2024

@author: siets009
"""
import pandas as pd
import numpy as np

import geopandas as gpd

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set_style("white")


#%%#Load data & merge
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
print(world.crs)
df = pd.read_csv("world_map_geopandas.csv")
gdf = pd.merge(world, df, left_on = 'iso_a3', right_on = 'Alpha-3')
#Not all countries are in the default map, however...
print("Missing:")
print(set(df['Country']) - set(gdf['Country']))
#... these are basically all 

#%% Quick plot /w geopandas built-in function
gdf.plot(column="Nr of datasets")

#%% Nicer plot

#Replace 0s with nan
gdf.replace(0, np.nan, inplace=True)
#Remove Antartica
gdf = gdf[gdf['name'] != 'Antarctica']

#Change to Eckert IV projection
gdf = gdf.to_crs("+proj=eck4 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")
print("\nNew projection set:")
print(gdf.crs)

# Plot country borders
fig, ax = plt.subplots(figsize=(14,10), dpi=300)
ax = gdf["geometry"].boundary.plot(ax=ax, edgecolor='grey', linewidth=1)

#Set a discrete colorbar
cmap = mpl.cm.viridis_r
bounds = [4,5,6,7,8]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='min')


#Plot chloropleth
gdf.plot(column="Nr of datasets", ax=ax, cmap='viridis_r', 
         legend=False, #Add custom later
         vmin=bounds[0]-1, #To make sure they match up (note the extend=min)
         missing_kwds={"color": "lightgrey", "edgecolor": "white"})
#Add colorbar
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             ax=ax, orientation='horizontal',
             label="Nr of indices/datasets", 
             shrink = 0.5, pad = 0.01)

#ax.set_title("Coverage of non-sectoral adaptation-relevant datasets", weight='bold', size=16)

#Remove all spines/axes labels
sns.despine(ax = ax, top=True, bottom=True, left=True, right=True)
ax.set_xticks([])
ax.set_yticks([])

fig.tight_layout()

fig.savefig(r'plots/country_coverage.png', bbox_inches='tight', dpi=300)
