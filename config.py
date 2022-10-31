import os
import datetime

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

DATA_DIR = os.path.join(ROOT_DIR, 'data')
DATA_IN = os.path.join(DATA_DIR, 'in')
DATA_OUT = os.path.join(DATA_DIR, 'out')

SCINTILLOMETER_DIR = os.path.join(DATA_IN, 'scintillometer')
SCINTILLOMETER_DIR_RAW = os.path.join(SCINTILLOMETER_DIR, 'raw')
SCINTILLOMETER_DIR_PROCESSED = os.path.join(SCINTILLOMETER_DIR, 'processed')

WEATHER_DIR = os.path.join(DATA_IN, 'weather')
WEATHER_DIR_RAW = os.path.join(WEATHER_DIR, 'raw')
WEATHER_DIR_PROCESSED = os.path.join(WEATHER_DIR, 'processed')

NDBC_DIR = os.path.join(DATA_IN, 'ndbc')

DOCS_DIR = os.path.join(ROOT_DIR, 'docs')
FIGS_DIR = os.path.join(DATA_DIR, 'figures')
UTILS_DIR = os.path.join(ROOT_DIR, 'utils')
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts')

MODELS_DIR = os.path.join(DATA_DIR, 'models')
CHPK_DIR = os.path.join(MODELS_DIR, 'checkpoints')
LOGS_DIR = os.path.join(MODELS_DIR, 'logs')

DLST = {
    "2019": [
        datetime.datetime.fromisoformat('2019-03-10T02:00:00'),
        datetime.datetime.fromisoformat('2019-11-03T02:00:00'),
        datetime.timedelta(hours=1)
    ],
    "2020": [
        datetime.datetime.fromisoformat('2020-03-08T02:00:00'),
        datetime.datetime.fromisoformat('2020-11-01T02:00:00'),
        datetime.timedelta(hours=1)
    ],
    "2021": [
        datetime.datetime.fromisoformat('2021-03-14T02:00:00'),
        datetime.datetime.fromisoformat('2021-11-07T02:00:00'),
        datetime.timedelta(hours=1)
    ],
    "2022": [
        datetime.datetime.fromisoformat('2022-03-13T02:00:00'),
        datetime.datetime.fromisoformat('2022-11-06T02:00:00'),
        datetime.timedelta(hours=1)
    ],
}
LONG_DEG = '76.479W'  # USNA Severn-River Scintillometer lat-long
LAT_DEG = '38.983N'
DEG_EW = float(LONG_DEG[:-1])  # float (numerical) long in decimal degrees
if LONG_DEG[-1] == 'W':
    DEG_EW *= -1
DEG_NS = float(LAT_DEG[:-1])  # float (numerical) lat in decimal degrees
if LAT_DEG[-1] == 'S':
    DEG_NS *= -1
TZ_OFFSET = -5
DEFAULT_START_TIME = datetime.datetime.fromisoformat('2020-01-01T00:00:00')
OBS_LOCATION = 'Annapolis'
OBS_COUNTRY = 'USA'
OBS_TZ = 'America/New_York'
WRITE_CSV = True
DEBUG = True
