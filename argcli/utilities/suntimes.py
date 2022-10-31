import datetime
import math

import pandas as pd
from astral import LocationInfo
from astral.sun import sun

from config import DLST, TZ_OFFSET, DEG_NS, DEG_EW, OBS_LOCATION, OBS_COUNTRY, OBS_TZ

OBS_LOC = LocationInfo(OBS_LOCATION, OBS_COUNTRY, OBS_TZ, DEG_NS, DEG_EW)


def _add_daylight_savings(df: pd.DataFrame,
                          dlst: dict = DLST,
                          column: str = 'timestamp') -> pd.DataFrame:
    """Adjusts time timestamps in the DataFrame to conform with DLST.

    Using the dictionary of supplied daylight savings start and end
    timestamps, this method adjusts records within the daylight
    savings interval such that all are in Eastern time, without 
    forcing a timezone on the column. Duplicate timestamps are
    removed with a preference for the last time stamp.

    Args:
        df: The path to the scintillometer data directory containing raw
          data measurements.
        dlst: The dictionary of records for daylight savings start
          and end timestamps, along with the time adjustment of the form
          [timestamp_start, timestamp_end, hour_change]

    Returns:
        A modified Pandas DataFrame with adjustments made to timestamps.

        The DataFrame is sorted by Timestamp, in monotonically increasing
          order.
    """
    for k, v in dlst.items():
        time_start = v[0]
        time_end = v[1]
        hour_change = v[2]
        df.loc[((df[column] >= time_start) & (df[column] <= time_end)),
               'timestamp'] = df.loc[(
                   (df[column] >= time_start) &
                   (df[column] <= time_end)), 'timestamp'] + hour_change

    df = df.groupby(column).last()
    return df.reset_index()


def _get_sunrise_sunset_for_datetime(
        dt: datetime.datetime) -> datetime.datetime:
    s = sun(OBS_LOC.observer, date=dt)
    return (s['sunrise'], s['sunset'])


def add_sunrise_sunset(df: pd.DataFrame) -> pd.DataFrame:
    df['date'] = pd.to_datetime(df.timestamp).dt.date
    df['time_sunrise'] = df['date'].apply(
        lambda x: _get_sunrise_sunset_for_datetime(x)[0])
    df['time_sunset'] = df['date'].apply(
        lambda x: _get_sunrise_sunset_for_datetime(x)[1])
    df['time_sunrise'] = df['time_sunrise'].dt.tz_convert('US/Eastern')
    df['time_sunrise'] = df['time_sunrise'].dt.tz_localize(tz=None)
    df['time_sunset'] = df['time_sunset'].dt.tz_convert('US/Eastern')
    df['time_sunset'] = df['time_sunset'].dt.tz_localize(tz=None)
    return df
