# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 12:03:36 2024

@author: siets009
"""
from time import time
import pandas as pd
import numpy as np
import os

#plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib import colormaps
import seaborn as sns

#Load - separate excel with only relevant columns
#DATA_DIR = r'C:\Users\siets009\OneDrive - Wageningen University & Research\Indices'
#DATA_DIR = r"D:\OneDrive - Wageningen University & Research\Indices"
DATA_DIR = r'C:\Users\ajsie\OneDrive - Wageningen University & Research\Indices'
df = pd.read_excel(os.path.join(DATA_DIR, 'Heatmap.xlsx'))

df.rename(columns = {
    'Impact, risk & vulnerability': 'Impact, risk &\n vulnerability',
    'Monitoring, evaluation & learning': 'Monitoring, evaluation\n& learning',
    'Means of implementation': 'Means of\nimplementation',
    'Infrastructure & settlements': 'Infrastructure\n& settlements'
    }, inplace = True)

#TODO: filter out "not found"? We didn't rate them so should be automatic, but cleaner in a sense
#%%
def columns_to_count_matrix(df,
                            columns_out,
                            rows_out,
                            match_on = 'Yes',
                            non_exact = True,
                            multiply_col = False):
    
    
    #Store the nr of matches in a dict of dicts
    cols = {}
    for col in columns_out:
        rows = {}
        for row in rows_out:
            #Temporary df where row and column both match
            if non_exact:
                tdf = df[(df[col].str.contains(match_on, case=False)) &(
                        df[row].str.contains(match_on, case = False))]
            else:
                tdf = df[(df[col] == match_on) &(
                        df[row] == match_on)]
            #We want to record the nr of matches    
            n = len(tdf)
            #If we want to scale values by a value, multiply by sum of that column
            if multiply_col:
                n = n*tdf[multiply_col].sum()
            #note this in the dict
            rows[row] = n
            if n == 0:
                print(f"NB: no results for {row} x {col}")
        #Add all rows to dict of dicts
        cols[col] = rows
    #Return as a dataframe
    return(pd.DataFrame.from_dict(cols))

def plot_heatmap(df_count, 
                 fig = None, ax = None,
                 title = None,
                 sort_by_col_size = True,
                 col_normalise = False, log_normalise=False,
                 add_total = True):
    
    #Sort the columns in order of magnitude
    if sort_by_col_size:    
        df_count = df_count[df_count.sum(0).sort_values(ascending=False).index]
    
    #Colours normalised by column -- else, keep absolute
    if col_normalise:
        df_norm = 100*df_count/df_count.sum()
        annot = df_count
        cbar_format = '%.0f%%'
        cbar_label = 'Share of indices/datasets in column'
    else:
        df_norm = df_count #df_norm gets plotted so needs to be defined
        annot = True
        cbar_format = '%.0f'
        cbar_label = 'Number of indices/datasets'
        
    if log_normalise:
        norm = LogNorm()
    else:
        norm = None
        
    #Add a total row with absolute numbers
    if add_total:
        #Add a total row and column
        df_count['Total number'] = df_count.sum(axis = 1)
        df_count.loc['Total number'] = df_count.sum(axis = 0)
        
        #Now put the counts in a separate df with all NaN except the rows we added
        df_tot = df_count.copy()
        df_tot.iloc[:-1, :-1] = np.nan
        #The total of totals should also not be plotted
        df_tot.loc['Total number', 'Total number'] = np.nan
        
        #Inverse so the total row does not get plotted with color in the normalised df
        df_norm['Total number'] = np.nan
        df_norm.loc['Total number'] = np.nan
        
        annot = df_count
        
        #pdb.set_trace()
    
    #Start plot
    if fig == None:
        fig, ax = plt.subplots(1,1, figsize = (10, 10), dpi=200)
    
    #Plot blue to yellow on a white background
    cmap = colormaps.get_cmap('viridis_r')
    cmap.set_bad("w")
    
    ax = sns.heatmap(df_norm, annot=annot, linewidth=.01,
                     norm=norm, 
                     xticklabels=True, yticklabels=True, #Force to display all labels
                     fmt='.0f',
                     mask = df_count <0,
                     linecolor = 'w',
                     square=True, #Force cells to be square shaped => does mess with fig dimensions
                     vmin = 1,
                     cmap = cmap, cbar_kws={"shrink": 0.4,
                                            "pad": 0.05,
                                            'format': cbar_format,
                                            #"extend": 'both',
                                            'label': cbar_label})
    if add_total:
        #Plot the total values (ony final row and column are not NaN)
        ax = sns.heatmap(df_tot, ax = ax, linewidth=0,
                         annot = True, fmt = '.0f',
                         annot_kws={"style": "italic", "weight": "bold"},
                         cbar = False,
                         #Make the total row white (max of inverted grey range)
                         vmax = -1, vmin = -2,
                         cmap =  colormaps.get_cmap('Greys_r'), #norm=LogNorm(),
                         mask = df_tot.isna())
    
    plt.xticks(rotation=45, ha='right')
    
    if title  != None:
        ax.set_title(title, weight='bold')
        
    plt.tight_layout()
    return(fig, ax)


#%% Most basic: try all datasets
col_columns = ['Water & sanitation', 'Food & agriculture', 'Health', 'Ecosystems',
               'Infrastructure\n& settlements', 'Livelihoods', 'Cultural heritage',
               #'Not in framework' 
               ]
row_columns = ['Impact, risk &\n vulnerability', 'Planning',
               'Implementation', 'Monitoring, evaluation\n& learning',
               'Means of\nimplementation']

df_count = columns_to_count_matrix(df, col_columns, row_columns)

sns.set_style("whitegrid", {'axes.grid' : False})

fig, ax = plot_heatmap(df_count,
                       title = None)

#Add vertical line to separate Means of implementation visually
ax.axhline(4, c='w', lw=7)

#In this case, top labels seems to look better
plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
plt.xticks(rotation=315, ha='right')

fig.tight_layout()

#%% Only climate-specific

df_count_s = columns_to_count_matrix(df[df['Climate specific'].isin(['Yes', 'Seperable'])], 
                                     col_columns, row_columns)

fig2, ax2 = plot_heatmap(df_count_s,
                       title = "Number of climate-specific datasets covering theme and part of policy cycle")

#%% Multiply by nr of times cited
df_count_m = columns_to_count_matrix(df, col_columns, row_columns,
                                     multiply_col = 'Times cited')

fig2, ax2 = plot_heatmap(df_count_m,
                         log_normalise=False,
                         title = "Scientific importance of climate-specific datasets covering theme and part of policy cycle")

#TODO: separate out means of implementation
#TODO: check titles
#TODO: total n
