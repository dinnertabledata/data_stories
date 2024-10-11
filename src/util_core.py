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
