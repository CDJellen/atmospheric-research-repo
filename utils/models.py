import pandas as pd
import numpy as np


def sadotMM_Pred(df):
    """
    Model 1 from Sadot et el: Forecasting Optical Turbulence Strength
    :param df:
    :return:
    """
    T = df.Air_Temperature.values  #Temperature in C
    T = T + 273.15
    U = df.Wind_Speed.values  #Wind Speed in m/s
    #U_Gust = df_plot.Wind_Speed_Gust.values #Wind Speed in m/s
    RH = df.Relative_Humidity.values  #Relative Humidity Percent
    W = df.Temporal_Hour_Weight_Weighted.values  #Temporal Hour Weight
    #W = W/12

    a1 = -1.58e-15
    b1 = 2.74e-16
    c1 = 8.30e-17
    c2 = -2.22e-18
    c3 = 1.42e-20
    d1 = 3.37e-16
    d2 = 1.92e-16
    d3 = -2.8e-17
    e = -7.44e-14

    mdl_2_pred_Cn2 = (a1 * W + b1 * T + c1 * RH + c2 * (RH * RH) + c3 *
                      (RH * RH * RH) + d1 * (U) + d2 * (U * U) + d3 *
                      (U * U * U) + e) * ((3 / 15)**(-2 / 3))

    return (mdl_2_pred_Cn2[:])


def sadotOMM_Pred(df):
    """
    Offshore Macro Meterological Model from Sadot et el: Forecasting Optical Turbulence Strength
    :param df:
    :return:
    """
    T = df.Air_Temperature.values  #Temperature in C
    T = T + 273.15
    U = df.Wind_Speed.values  #Wind Speed in m/s
    #U_Gust = df_plot.Wind_Speed_Gust.values #Wind Speed in m/s
    RH = df.Relative_Humidity.values  #Relative Humidity Percent
    W = df.Temporal_Hour_Weight.values  #Temporal Hour Weight
    #W = W/12

    a1 = -1.58e-15
    b1 = 2.74e-16
    c1 = 8.30e-17
    c2 = -2.22e-18
    c3 = 1.42e-20
    d1 = 3.37e-16
    d2 = 1.92e-16
    d3 = -2.8e-17
    e = -7.44e-14

    mdl_2_pred_Cn2 = (a1 * W + b1 * T + c1 * RH + c2 * (RH * RH) + c3 *
                      (RH * RH * RH) + d1 * (U) + d2 * (U * U) + d3 *
                      (U * U * U) + e) * ((3 / 15)**(-2 / 3))
    #mdl_2_pred_Cn2_Gust = a1*W + b1*T + c1*RH + c2*(RH*RH) + c3*(RH*RH*RH) + d1*(U_Gust) + d2*(U_Gust*U_Gust) + d3*(U_Gust*U_Gust*U_Gust) + e

    return (mdl_2_pred_Cn2[:])


def oermann_2014_pred(df):
    """
    Offshore Macro Meterological Model from Sadot et el: Forecasting Optical Turbulence Strength
    :param df:
    :return:
    """
    T_gnd = df.Air_Water_Temperature_Difference.values * (-1)  #Temperature in C
    T_air = df.Air_Temperature.values * 0.05  # temperature in C
    WS = df.Wind_Speed.values  #Wind Speed in m/s
    RH = df.Relative_Humidity.values  #Relative Humidity Percent
    SF = df.Solar_Radiation.values  #Temporal Hour Weight

    a1 = -1.58e-15
    b1 = 2.74e-16
    c1 = 8.30e-17
    c2 = -2.22e-18
    c3 = 1.42e-20
    d1 = 3.37e-16
    d2 = 1.92e-16
    d3 = -2.8e-17
    e = -3.57e-13

    mdl_2_pred_Cn2 = e + (
        (-2.95e-14) * T_air + (5.20e-15) * (T_air * T_air) + (-9.99e-17) *
        (T_air * T_air * T_air)) + (
            (-1.45e-16) * SF + 1.36e-18 * SF * SF - 7.21e-22 * SF * SF * SF
        ) + (7.41e-14 * WS - (1.08e-14) * WS * WS + 3.71e-16 * WS * WS * WS) + (
            1.50e-13 * T_air + 4.11e-13 * T_air * T_air -
            5.69e-13 * T_air * T_air * T_air) + (
                1.44e-14 * RH - 1.86e-16 * RH * RH + 6.64e-19 * RH * RH * RH)

    return (mdl_2_pred_Cn2[:])**(4 / 3)


def wangMMM_Pred(df):
    """
    Wang Modified Macro Meterological Model Coastal
    :param df:
    :return:
    """
    T = df.Air_Temperature.values  # Temperature in C
    U = df.Wind_Speed.values  # Wind Speed in m/s
    RH = df.Relative_Humidity.values  # Relative Humidity Percent
    W = df.Temporal_Hour_Weight.values  # Temporal Hour Weight
    W = W * 12

    # constant coefficient
    e = 4963.45

    # wind speed parameters
    a1 = -12.1635
    a2 = 1.39064
    a3 = -0.167931
    a4 = 0.00869613

    # temperature parameters
    b1 = -265.847
    b3 = 0.213973
    b4 = -0.00311029

    # relative humidity parameters
    c1 = -32.7117
    c3 = -0.00315137
    c2 = 1.38747e-5

    # wind speed temperature cross parameters
    ab1 = 0.200515
    ab2 = -0.000268105

    # wind speed relative humidity cross parameters
    ac1 = 0.0585552
    ac2 = -1.55805e-5

    # temperature relative humidity cross parameters
    bc1 = 0.632929
    bc2 = 0.0229636
    bc3 = -0.000429867

    mdl_3_pred_Cn2 = 1e-14 * (e + (a1 * U) + (b1 * T) + (c1 * RH) +
                              (a2 * U * U) + (ab1 * U * T) + (ac1 * RH * U) +
                              (bc1 * T * RH) + (a3 * U * U * U) +
                              (b3 * T * T * T) + (bc2 * T * RH * RH) +
                              (c3 * RH * RH * RH) + (a4 * U * U * U * U) +
                              (ab2 * U * U * T * T) + (ac2 * U * U * RH * RH) +
                              (b4 * T * T * T * T) + (bc3 * T * T * RH * RH) +
                              (c2 * RH))

    return (mdl_3_pred_Cn2[:])


def rajEQ1_Pred(df):
    """
    EQ-1 from Raj et. al.
    :param df:
    :return:
    """
    T = df.Air_Temperature.values  # Temperature in C
    RH = df.Relative_Humidity.values  # Relative Humidity Percent
    W = df.Temporal_Hour_Weight.values  # Temporal Hour Weight
    U = df.Wind_Speed.values  # Wind Speed in m/s

    # constant coefficient
    e = 399.774

    # air temperature parameters
    a1 = -14.7804
    a3 = 0.00276336
    ab1 = 0.203989
    ac1 = 0.153508
    ac2 = 0.000116384

    # wind speed parameters
    b1 = 4.88372
    b2 = -2.32469
    b3 = 0.170863
    bc1 = -0.0616557

    # relative humidity parameters
    c1 = -5.58958
    c3 = 9.30618e-5

    # visibility parameters
    # d1 = 0.008

    # seasonality parameters
    e1 = -0.071

    # temporal hour weight parameters
    f1 = -2.46

    mdl_4_pred_Cn2 = 1e-14 * (e + U * b1 + T * a1 + RH * c1 + U * U * b2 +
                              ab1 * U * T + bc1 * U * RH + ac1 * T * RH +
                              b3 * U * U * U + a3 * T * T * T +
                              ac2 * T * RH * RH + c3 * RH * RH * RH)
    return (mdl_4_pred_Cn2[:])


def BKB_pred(df):
    """
    Bendrersky, Kopeika, Blaustein Model(Oermann 2014) sources 130 and 65
    :param df:
    :return:
    """
    T = df.Air_Temperature.values  # Temperature in C
    RH = df.Relative_Humidity.values  # Relative Humidity Percent
    W = df.Temporal_Hour_Weight_Weighted.values  # Temporal Hour Weight
    U = df.Wind_Speed.values  # Wind Speed in m/s

    # constant coefficient
    e = -4.45e-14

    # air temperature parameters
    a = 0.35e-4

    # wind speed parameters
    b1 = 2.58e-14  # assume grassland

    # relative humidity parameters
    c1 = -6.797e-15

    # temporal hour weight parameters
    w = 3.8e-14

    mdl_4_pred_Cn2 = w * W + a * exp((-1) * T) + b1 * U + c1 * RH + e
    return (mdl_4_pred_Cn2[:])


def BKB_2_pred(df):
    """
    Bendrersky, Kopeika, Blaustein Model (Oermann 2014) sources 130 and 65
    :param df:
    :return:
    """
    T = df.Air_Temperature.values  # Temperature in C
    RH = df.Relative_Humidity.values  # Relative Humidity Percent
    W = df.Temporal_Hour_Weight_Weighted.values  # Temporal Hour Weight
    U = df.Wind_Speed.values  # Wind Speed in m/s

    # constant coefficient
    e = -4.6e-13

    # air temperature parameters
    a = 2e-15

    # wind speed parameters
    b1 = -2.5e-15
    b2 = 1.2e-15
    b3 = -8.5e-17

    # relative humidity parameters
    c1 = -2.8e-15
    c2 = 2.9e-17
    c3 = -1.1e-19

    # temporal hour weight parameters
    w = 3.8e-14

    mdl_4_pred_Cn2 = w * W + a * T + (b1 * U + b2 * U * U + b3 * U * U * U) + (
        c1 * RH + c2 * RH * RH + c3 * RH * RH * RH)
    return (mdl_4_pred_Cn2[:])
