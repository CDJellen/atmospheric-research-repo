import os
import glob

import pandas as pd
import numpy as np

from config import WEATHER_DIR_RAW, WEATHER_DIR_PROCESSED
from argcli.data._base_handler import BaseDataHandler


class LocalWeatherDataHandler(BaseDataHandler):

    expected_data_path = os.path.join(WEATHER_DIR_PROCESSED, 'weather.csv')

    @classmethod
    def load(cls, debug_: bool = BaseDataHandler.debug_) -> pd.DataFrame:
        """Reads processed weather data from csv.

        Retrieves processed weather records and return them as a pandas
        DataFrame.

        Args:
            debug_: Whether to log progress.

        Returns:
            A Pandas DataFrame object with two columns, the timestamp and the 
            Cn2 reading produced by the local weather station. The DataFrame is
            sorted by Timestamp, in monotonically increasing order.

        Raises:
            IOError: An error occurred accessing any file in the supplied dir.
        """
        if os.path.exists(LocalWeatherDataHandler.expected_data_path):
            try:
                df = pd.read_csv(LocalWeatherDataHandler.expected_data_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            except Exception as e:
                raise IOError from e
        else:
            return read_data(return_df=True)


def read_data(fp_in: str = WEATHER_DIR_RAW,
              fp_out: str = WEATHER_DIR_PROCESSED,
              return_df: bool = True,
              write_csv: bool = BaseDataHandler.write_csv,
              debug_: bool = BaseDataHandler.debug_) -> pd.DataFrame:
    """Reads weather records to Pandas DataFrame.

    Retrieves weather records from Excel workbooks and reads them
    into a Pandas DataFrame, appending all records to a single DataFrame,
    sorting the records by timestamp, and returning the DataFrame.

    Args:
        fp_in: The path to the weather station data directory containing raw
        data measurements.
        fp_out: The path to the weather station data directory at which to
        optionally write weather station data as a csv.
        return_df: Whether to return the DataFrame.
        write_csv: Whether to serialize the DataFrame to disk.
        debug_: Whether to log progress.

    Returns:
        A Pandas DataFrame object with two columns, the timestamp and the
        readings produced by the weather station. The DataFrame is sorted by
        Timestamp, in monotonically increasing order.

    Raises:
        IOError: An error occurred accessing any file in the supplied dir.
    """
    df_weather = _get_df_from_records_weather(fp_in=fp_in, debug_=debug_)

    if write_csv:
        df_weather.to_csv(os.path.join(fp_out, 'weather.csv'), index=False)
    if return_df:
        return df_weather
    else:
        del df_weather


"""PRIVATE"""


def _get_df_from_records_weather(fp_in: str, debug_: bool) -> pd.DataFrame:
    """Reads Davis weather station records to Pandas DataFrame.

    Retrieves weather records from Excel workbooks and reads them
    into a Pandas DataFrame, appending all records to a single DataFrame, 
    sorting the records by timestamp, and returning the DataFrame.

    Args:
        fp_in: The path to the scintillometer data directory containing raw
        data measurements.

    Returns:
        A Pandas DataFrame object with two columns, the timestamp and the 
        Cn2 reading produced by the scintillometer.

        The DataFrame is sorted by Timestamp, in monotonically increasing
        order.

    Raises:
        IOError: An error occurred accessing any file in the supplied dir.
    """
    df_weather = None
    try:
        for name in glob.glob(str(os.path.join(fp_in, '*.xlsx'))):
            if debug_:
                print(name)
            if df_weather is None:
                df_weather = pd.read_excel(name,
                                           skiprows=range(2),
                                           parse_dates=[[0, 1]],
                                           header=None,
                                           na_values={'---', '------'},
                                           dtype={
                                               '2:7': np.float64,
                                               '8': str,
                                               '9:10': np.float64,
                                               '11': str,
                                               '12:37': np.float64
                                           })  #dtype={'Cn2': np.float64}
            else:
                df_weather = df_weather.append(
                    pd.read_excel(name,
                                  skiprows=range(2),
                                  parse_dates=[[0, 1]],
                                  header=None,
                                  na_values={'---', '------'},
                                  dtype={
                                      '2:7': np.float64,
                                      '8': str,
                                      '9:10': np.float64,
                                      '11': str,
                                      '12:37': np.float64
                                  }))
        df_weather.columns = [
            'timestamp', 'temperature_air', 'temperature_air_high',
            'temperature_air_low', 'humidity', 'dew_point_temperature',
            'wind_speed', 'wind_direction', 'wind_run', 'wind_speed_high',
            'wind_direction_high', 'wind_chill', 'heat_index', 'thw_index',
            'thsw_index', 'pressure', 'rain', 'rain_rate', 'solar_radiation',
            'solar_energy', 'solar_radiation_high', 'uv_index', 'uv_dose',
            'uv_high', 'heat_dd', 'cool_dd', 'inner_temperature',
            'inner_humidity', 'inner_dew_point_temperature', 'inner_heat',
            'inner_emc', 'air_density', 'evapotranspiration', 'wind_samp',
            'wind_tx', 'iss_recept', 'arc_int'
        ]
        df_weather['timestamp'] = pd.to_datetime(df_weather['timestamp'])
        df_weather.sort_values(by='timestamp', inplace=True)
        return df_weather
    except Exception as e:
        raise IOError from e
