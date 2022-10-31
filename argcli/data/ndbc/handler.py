import os
from typing import Union

import pandas as pd
from ndbc_api import NdbcApi

from config import LONG_DEG, LAT_DEG, NDBC_DIR, DEFAULT_START_TIME
from argcli.data._base_handler import BaseDataHandler


class NdbcDataHandler(BaseDataHandler):

    expected_data_path = os.path.join(NDBC_DIR, 'data_buoy.csv')

    @classmethod
    def load(cls, debug_: bool = BaseDataHandler.debug_) -> pd.DataFrame:
        """Reads processed NDBC buoy data from csv.

        Retrieves processed water temperature records from ndbc buoy data and
        returns them as a pandas DataFrame.

        Args:
            debug_: Whether to log progress.

        Returns:
            A Pandas DataFrame object with one column, `temperature_water`,
            indexed by timestamp. The timestamp is unique and sorted in
            monotonically increasing order.

        Raises:
            IOError: An error occurred accessing any file in the supplied dir.
        """
        if os.path.exists(NdbcDataHandler.expected_data_path):
            try:
                df = pd.read_csv(NdbcDataHandler.expected_data_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            except Exception as e:
                raise IOError from e
        else:
            return read_data(return_df=True, debug_=debug_)


def read_data(fp_in: Union[str, None] = None,
              fp_out: str = NDBC_DIR,
              return_df: bool = True,
              write_csv: bool = BaseDataHandler.write_csv,
              debug_: bool = BaseDataHandler.debug_) -> pd.DataFrame:
    """Reads NDBC data buoy records to Pandas DataFrame.

    Retrieves ndbc records using the `ndbc_api.NdbcApi` and reads them
    into a Pandas DataFrame, appending all records to a single DataFrame,
    sorting the records by timestamp, and returning the DataFrame.

    Args:
        fp_in: The path to the ndbc data directory containing raw
            buoy measurements, if implemented. The default makes a one-time
            query against the NDBC data service using the `ndbc_api.NdbcApi`.
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
        NotImplementedError: An `fp_in` was supplied.
    """
    if not fp_in:
        df_ndbc = _get_df_from_records_ndbc(fp_in=fp_in, debug_=debug_)
    else:
        raise NotImplementedError

    if write_csv:
        df_ndbc.to_csv(os.path.join(fp_out, 'data_buoy.csv'), index=False)
    if return_df:
        return df_ndbc
    else:
        del df_ndbc


"""PRIVATE"""


def _get_df_from_records_ndbc(fp_in: Union[str, None],
                              debug_: bool) -> pd.DataFrame:
    """Reads NDBC data buoy records to Pandas DataFrame.

    Retrieves buoy records from the NDBC data service and reads them into a
    pandas DataFrame, appending all records to a single DataFrame, sorting
    the records by timestamp, and returning the DataFrame.

    Args:
        fp_in: The path to the scintillometer data directory containing raw
            data measurements.
        debug_: Whether to log progress.

    Returns:
        A Pandas DataFrame object data columns, the timestamp and the buoy
        readings produced by the scintillometer. The DataFrame is sorted by
        Timestamp, in monotonically increasing order.

    Raises:
        IOError: An error occurred accessing any file in the supplied dir.
        NotImplementedError: An `fp_in` was supplied.
    """
    if fp_in:
        raise NotImplementedError
    api = NdbcApi(cache_limit=1000)
    try:
        if debug_:
            NdbcDataHandler.logger.debug(
                f'Querying the NDBC using a {type(api)} at {id(api)}')
        station_name = api.nearest_station(lat=LAT_DEG, lon=LONG_DEG)
        if debug_:
            NdbcDataHandler.logger.debug(
                f'Nearest station to {LAT_DEG}, {LONG_DEG} is {station_name}')
        df_ndbc = api.get_data(station_id=station_name,
                               mode='stdmet',
                               start_time=DEFAULT_START_TIME)
        df_ndbc.index = df_ndbc.index.tz_localize(tz='UTC')
        df_ndbc.index = df_ndbc.index.tz_convert('US/Eastern')
        df_ndbc.index = df_ndbc.index.tz_localize(tz=None)
        df_ndbc.rename(columns={'WTMP': 'temperature_water'}, inplace=True)
        df_ndbc.reset_index(inplace=True)
        df_ndbc.sort_values(by='timestamp', inplace=True)
        df_ndbc = df_ndbc.drop_duplicates(subset=['timestamp'], keep='last')
        df_ndbc['timestamp'] = pd.to_datetime(df_ndbc['timestamp'])
        return df_ndbc[['timestamp', 'temperature_water']]
    except Exception as e:
        raise IOError from e
