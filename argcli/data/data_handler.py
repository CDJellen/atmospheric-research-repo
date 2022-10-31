import os

import pandas as pd
import numpy as np

from config import DATA_IN, DATA_OUT, MODELS_DIR, LOGS_DIR


SCINTILLOMETER_DIR = os.path.join(DATA_IN, 'scintillometer')
SCINTILLOMETER_DIR_RAW = os.path.join(SCINTILLOMETER_DIR, 'raw')
SCINTILLOMETER_DIR_PROCESSED = os.path.join(SCINTILLOMETER_DIR, 'processed')

WEATHER_DIR = os.path.join(DATA_IN, 'weather')
WEATHER_DIR_RAW = os.path.join(WEATHER_DIR, 'raw')
WEATHER_DIR_PROCESSED = os.path.join(WEATHER_DIR, 'processed')

NDBC_DIR =  os.path.join(DATA_IN, 'ndbc')

