import pandas as pd
import numpy as np
import datetime as dt
import os

from sklearn import preprocessing


from config import ROOT_DIR, DATA_DIR, DATA_IN, DATA_OUT, FIGS_DIR, MODELS_DIR

def get_cn2(data_dir=DATA_IN, dropna=True, drop_above=5e-12):
    """
    <docstring>
    """
    df_sc = pd.read_excel(os.path.join(DATA_IN, "Cn2_Data_Scintillometer.xlsx"), skiprows = np.arange(101), header = None, parse_dates = True,dtype={'A':np.datetime64,"B":np.float32}, engine='openpyxl')
    df_sc.columns = ['Timestamp','Cn2']
    if dropna:
        df_sc = df_sc.dropna()
    df_sc = df_sc.loc[df_sc['Cn2']<drop_above]
    df_sc_dlst_mask = (df_sc['Timestamp'] >= dt.datetime(2020, 3, 8, 2))&(df_sc['Timestamp'] <= dt.datetime(2020, 11, 1, 2))
    df_sc['Timestamp'].loc[df_sc_dlst_mask] = df_sc['Timestamp'].loc[df_sc_dlst_mask] + dt.timedelta(hours=1)
    df_sc = df_sc.set_index('Timestamp')
    return df_sc

def get_wx(data_dir=DATA_IN, dropna=True, skiprows=np.arange(2), usecols="A,D,G:M,R,U,X,AH", na_values={'---', '------'}):
    """
    <doctring>
    """
    df_wx = pd.read_excel(os.path.join(DATA_IN, "01_JAN_to_15JUNE2020_local_station_weather.xlsx"), engine='openpyxl', skiprows = skiprows, header= None, usecols = usecols, na_values=na_values, dtype={'A':np.datetime64,'C,F:H':np.float32,'I':np.int8,'J:K':np.float32,'L':np.int8,'Q':np.float32,'T':np.float32,'W':np.float32,'AG':np.float32})
    df_wx.columns = ["Timestamp", "Air_Temperature", "Relative_Humidity", "Dew_Point_Temperature", "Wind_Speed", "Wind_Direction","Wind_Run","Wind_Speed_Gust","Wind_Direction_Gust","Pressure", "Solar_Radiation", "UV_Index","Air_Density"]
    df_wx = df_wx.set_index('Timestamp')
    return df_wx

def get_ndbc(data_dir=DATA_IN, dropna=True, skiprows=np.arange(2), usecols="A:E,N,O", na_values=[99, 999]):
    """
    <docstring>
    """
    df_ndbc = pd.read_excel(os.path.join(DATA_IN, "NDBC-TPLM2-Data.xlsx"), header= None, na_values = na_values, usecols = usecols, skiprows = skiprows, dtype={'A:E':object,"N:O":np.float32}, names=['Year','Month','Day','Hour','Minute','Air_Temperature','Water_Temperature'],engine='openpyxl')
    df_ndbc['Air_Water_Temperature_Difference'] = df_ndbc['Air_Temperature']-df_ndbc['Water_Temperature']
    df_ndbc = df_ndbc.drop(columns=['Air_Temperature'])
    df_ndbc = df_ndbc.drop(columns=['Water_Temperature'])
    df_ndbc['Timestamp'] = df_ndbc['Year'].astype(str) + '/' + df_ndbc['Month'].astype(str) + '/' + df_ndbc['Day'].astype(str) + 'T' + df_ndbc['Hour'].astype(str) + ':' + df_ndbc['Minute'].astype(str)
    df_ndbc = df_ndbc.drop(columns=['Year','Month','Day','Hour','Minute'])
    df_ndbc['Timestamp'] = pd.to_datetime(df_ndbc['Timestamp'])
    if dropna:
        df_ndbc = df_ndbc.dropna()
    df_ndbc = df_ndbc.set_index("Timestamp")
    return df_ndbc

def get_sunrise(data_dir=DATA_IN, dropna=True):
    """
    <docstring>
    """
    df_sunrise = pd.read_excel(os.path.join(DATA_IN, "Annapolis_Sunrise_Sunset_Times_2020.xlsx"), header= None, parse_dates = [[0,1,2,3]],engine='openpyxl')
    df_sunrise.columns = ['Sunrise_Time','drop']
    df_sunrise = df_sunrise.drop(columns=['drop'])
    df_sunrise['Timestamp'] = (df_sunrise['Sunrise_Time'].dt.date).astype(np.datetime64)
    df_sunrise = df_sunrise.set_index("Timestamp")
    return df_sunrise

def get_sunset(data_dir=DATA_IN, dropna=True):
    """
    <docstring>
    """
    df_sunset = pd.read_excel(os.path.join(DATA_IN, "Annapolis_Sunrise_Sunset_Times_2020.xlsx"), header= None, parse_dates = [[0,1,2,4]],engine='openpyxl')
    df_sunset.columns = ['Sunset_Time','drop']
    df_sunset = df_sunset.drop(columns=['drop'])
    df_sunset['Timestamp'] = (df_sunset['Sunset_Time'].dt.date).astype(np.datetime64)
    df_sunset = df_sunset.set_index("Timestamp")
    return df_sunset

def enrich_temporal_hour_weight(df):
    """
    <docstring>
    """
    df['TH'] = (df['Sunset_Time'].values.astype(float) - df['Sunrise_Time'].values.astype(float))
    df['Temporal_Hour_Weight'] = (df['Timestamp'].values.astype(float) - df['Sunrise_Time'].values.astype(float))*12/(df['TH'])
    df['Solar_Hour'] = (df['Timestamp'].dt.hour - df['Sunrise_Time'].dt.hour)
    df['Temporal_Hour_Weight_Unscaled'] = df['Temporal_Hour_Weight']
    #make relative Temporal; Hour Weight
    val_TH = df['TH'].values
    val_TH = val_TH.reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    val_TH_scaled = min_max_scaler.fit_transform(val_TH)
    df['Temporal_Hour_Weight'] = val_TH_scaled[:]
    df = df.drop(columns=['TH'])
    return df

def enrich_temporal_hour_weight_oermann(df):
    """
    <docstring>
    """
    df['Solar_Hour_2'] = df['Solar_Hour']
    df['Solar_Hour_2'] = df['Solar_Hour_2'].where((df['Solar_Hour_2'] >= 0) & (df['Solar_Hour_2'] <= 16), 0)
    df['Temporal_Hour_Weight_Oermann'] = (df['Solar_Hour_2']*0.3257231 + (df['Solar_Hour_2'].pow(2))*0.01738 + (df['Solar_Hour_2'].pow(3))*-0.007102 + (df['Solar_Hour_2'].pow(4))*0.000305 - 0.5041482)
    df['Temporal_Hour_Weight_Oermann'] = df['Temporal_Hour_Weight_Oermann'].where(df['Temporal_Hour_Weight_Oermann'] >0.1,0.1)
    df = df.drop(columns=['Solar_Hour_2'])
    return df

def merge_frames(df_sc, df_wx, df_ndbc, df_sunrise, df_sunset):
    """
    <docstring>
    """
    df = df_sc.merge(df_wx, left_index=True, right_index=True, how='inner')
    df = df.merge(df_sunrise, left_index=True, right_index=True, how='outer').fillna(method='ffill')
    df = df.merge(df_sunset, left_index=True, right_index=True, how='outer').fillna(method='ffill')
    df = df.merge(df_ndbc, left_index=True, right_index=True, how='outer').fillna(method='ffill')
    df['Daylight_Time'] = df['Sunset_Time'] - df['Sunrise_Time']
    df = df.dropna()
    df = df.reset_index()
    return df

def merge_frames(df_sc, df_wx, df_ndbc, df_sunrise, df_sunset):
    """
    <docstring>
    """
    df = df_sc.merge(df_wx, left_index=True, right_index=True, how='inner')
    df = df.merge(df_sunrise, left_index=True, right_index=True, how='outer').fillna(method='ffill')
    df = df.merge(df_sunset, left_index=True, right_index=True, how='outer').fillna(method='ffill')
    df = df.merge(df_ndbc, left_index=True, right_index=True, how='outer').fillna(method='ffill')
    df['Daylight_Time'] = df['Sunset_Time'] - df['Sunrise_Time']
    df = df.dropna()
    df = df.reset_index()
    return df

def validate_df(df, log10=False):
    df['Air_Temperature'] = df['Air_Temperature'].astype(np.float64)
    df['Relative_Humidity'] = df['Relative_Humidity'].astype(int)
    df['Wind_Direction'] = df['Wind_Direction'].astype('category')
    df['Wind_Direction_Gust'] = df['Wind_Direction'].astype('category')
    df['Wind_Direction'], uniques_windDir = pd.factorize(df['Wind_Direction'])
    df['Wind_Direction_Gust'], uniques_windDirGust = pd.factorize(df['Wind_Direction_Gust'])
    df.astype({'Timestamp': 'datetime64[ns, US/Eastern]'})
    df['Month'] = df['Timestamp'].dt.month
    if log10:
        df['Log_Cn2'] = np.log10(df['Cn2'])
    df = df.reset_index(drop=True)
    return df

def get_df(data_dir=DATA_IN):
    """
    <docstring>
    """
    df_sc = get_cn2()
    df_wx = get_wx()
    df_ndbc = get_ndbc()
    df_sunrise = get_sunrise()
    df_sunset = get_sunset()
    df = merge_frames(df_sc, df_wx, df_ndbc, df_sunrise, df_sunset)
    df = enrich_temporal_hour_weight(df)
    df = enrich_temporal_hour_weight_oermann(df)
    df = validate_df(df, log10=True)
    return df
