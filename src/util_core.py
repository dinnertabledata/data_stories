#########################################################################
# 0. Imports

# General
import re, os, math, json
import itertools
from collections import Counter
from tqdm.notebook import tqdm
import difflib
import ast

# File IO
from pathlib import Path
import pickle
import json
import xlwings as xw

# Data
import pandas as pd
import numpy as np

# Data Viz
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


#########################################################################
# 1. General

def save_object(obj, file_path):
    directory = '/'.join(file_path.split('/')[:-1])
    Path(directory).mkdir(parents=True, exist_ok=True)
    with open(file_path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_object(file_path):
    with open(file_path, 'rb') as handle:
        obj = pickle.load(handle)
        return obj

def fix_json(text):
    # Stop at last complete item
    idx = text.rfind('}')
    text = text[:idx+1]
    # Add final curly bracket
    text = f'{text}\n}}'
    # Try to convert
    try: result = json.loads(text)
    except: result = None
    # Return
    return result

def string_similarity(str1, str2):
    result =  difflib.SequenceMatcher(a=str1.lower(), b=str2.lower())
    return result.ratio()


#########################################################################
# 2. Plotting

def plotly_scatter_v1(df, config):

    # Separate config
    name_col = config["name_col"]
    size_col = config["size_col"]
    color_col = config["color_col"]
    x_axis_col = config["x_axis_col"]
    y_axis_col = config["y_axis_col"]

    title = config["title"]
    x_axis_title = config["x_axis_title"]
    y_axis_title = config["y_axis_title"]

    min_point_size = config["min_point_size"]
    max_point_size = config["max_point_size"]
    x_axis_dtick = config["x_axis_dtick"]
    y_axis_dtick = config["y_axis_dtick"]
    width = config["width"]
    height = config["height"]

    # Create a new figure
    fig = go.Figure()

    # Scale sizes based on 'entries', ensuring they are between min_size and max_size
    scaled_sizes = ((df[size_col] - df[size_col].min()) / (df[size_col].max() - df[size_col].min())) * (max_point_size - min_point_size) + min_point_size

    # Normalize entries for coloring
    norm = plt.Normalize(df[color_col].min(), df[color_col].max())
    colors = plt.cm.terrain(norm(df[color_col]))  # Use 'terrain' colormap

    # Add traces for the data points
    for i in range(df.shape[0]):
        fig.add_trace(go.Scatter(
            x=[df[x_axis_col][i]],  # X value for each country
            y=[df[y_axis_col][i]],  # Y value for each country
            mode='markers+text',  # Show markers and text
            marker=dict(
                size=scaled_sizes[i],  # Use scaled sizes
                opacity=0.6,
                line=dict(width=1, color='Black'),
                color=f'rgb({int(colors[i][0]*255)}, {int(colors[i][1]*255)}, {int(colors[i][2]*255)})'  # Set color
            ),
            text=df[name_col][i],  # Display country name
            textposition='top center',  # Position text above the markers
            showlegend=False  # Hide legend
        ))

    # Update layout for better aesthetics
    fig.update_layout(
        title=title,
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        xaxis=dict(
            zeroline=True,
            zerolinecolor='Gray',
            zerolinewidth=2,
            title_standoff=10,
            dtick=x_axis_dtick,  # Major tick interval for x-axis
            gridcolor='LightGray',  # Add gridlines
            gridwidth=0.5
        ),
        yaxis=dict(
            zeroline=True,
            zerolinecolor='Gray',
            zerolinewidth=2,
            title_standoff=10,
            dtick=y_axis_dtick,  # Major tick interval for y-axis
            gridcolor='LightGray',  # Add gridlines
            gridwidth=0.5
        ),
        width=width,  # Set width to maintain square shape
        height=height,  # Set height to maintain square shape
    )

    # Show the plot
    fig.show()