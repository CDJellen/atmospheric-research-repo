import os
import glob
from typing import Union

import pandas as pd
import numpy as np

from config import SCINTILLOMETER_DIR_RAW, SCINTILLOMETER_DIR_PROCESSED
from argcli.data._base_handler import BaseDataHandler
from argcli.utilities import suntimes


class ScintillometerDataHandler(BaseDataHandler):

    expected_data_path = os.path.join(SCINTILLOMETER_DIR_PROCESSED,
                                      'scintillometer.csv')

    @classmethod
    def load(cls, debug_: bool = BaseDataHandler.debug_) -> pd.DataFrame:
        """Reads processed scintillometer data from csv.

        Retrieves processed scintillometer records and return them as a pandas
        DataFrame.

        Args:
            debug_: Whether to log progress.

        Returns:
            A Pandas DataFrame object with two columns, the timestamp and the 
            Cn2 reading produced by the scintillometer. The DataFrame is sorted 
            by Timestamp, in monotonically increasing order.

        Raises:
            IOError: An error occurred accessing any file in the supplied dir.
        """
        if os.path.exists(ScintillometerDataHandler.expected_data_path):
            try:
                df = pd.read_csv(ScintillometerDataHandler.expected_data_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            except Exception as e:
                raise IOError from e
        else:
            return read_data(return_df=True)


def read_data(
        fp_in: str = SCINTILLOMETER_DIR_RAW,
        fp_out: str = SCINTILLOMETER_DIR_PROCESSED,
        return_df: bool = True,
        write_csv: bool = BaseDataHandler.write_csv,
        debug_: bool = BaseDataHandler.debug_) -> Union[None, pd.DataFrame]:
    """Reads scintillometer records to Pandas DataFrame.

    Retrieves scintillometer records from Excel workbooks and reads them
    into a Pandas DataFrame, appending all records to a single DataFrame, 
    sorting the records by timestamp, and returning the DataFrame.

    Args:
        fp_in: The path to the scintillometer data directory containing raw
        data measurements.
        fp_out: The path to the scintillometer data directory at which to
        optionally write scintillometer data as a csv.
        return_df: Whether to return the DataFrame.
        write_csv: Whether to serialize the DataFrame to disk.
        debug_: Whether to log progress.

    Returns:
        A Pandas DataFrame object with two columns, the timestamp and the 
        Cn2 reading produced by the scintillometer. The DataFrame is sorted 
        by Timestamp, in monotonically increasing order.

    Raises:
        IOError: An error occurred accessing any file in the supplied dir.
    """
    df_sc = _get_df_from_records_scintillometer(fp_in=fp_in, debug_=debug_)
    df_sc = suntimes._add_daylight_savings(df_sc)

    if write_csv:
        df_sc.to_csv(os.path.join(fp_out, 'scintillometer.csv'), index=False)
    if return_df:
        return df_sc
    else:
        del df_sc


"""PRIVATE"""


def _get_df_from_records_scintillometer(fp_in: str, debug_: bool):
    """Reads scintillometer records to Pandas DataFrame.

    Retrieves scintillometer records from Excel workbooks and reads them
    into a Pandas DataFrame, appending all records to a single DataFrame, 
    sorting the records by timestamp, and returning the DataFrame.

    Args:
        fp_in: The path to the scintillometer data directory containing raw
        data measurements.
        debug_: Whether to log progress.

    Returns:
        A Pandas DataFrame object with two columns, the timestamp and the
        Cn2 reading produced by the scintillometer. The DataFrame is sorted
        by Timestamp, in monotonically increasing order.

    Raises:
        IOError: An error occurred accessing any file in the supplied dir.
    """
    df_sc = None
    try:
        for name in glob.glob(str(os.path.join(fp_in, '*.xlsx'))):
            if debug_:
                print(name)
            if df_sc is None:
                df_sc = pd.read_excel(name,
                                      parse_dates=[1],
                                      dtype={'Cn2': np.float64},
                                      header=[8])
            else:
                df_sc = df_sc.append(
                    pd.read_excel(name,
                                  parse_dates=[1],
                                  dtype={'Cn2': np.float64},
                                  header=[8]))
        df_sc['timestamp'] = pd.to_datetime(df_sc['timestamp'])
        df_sc.sort_values(by='timestamp', inplace=True)
        return df_sc
    except Exception as e:
        raise IOError from e
