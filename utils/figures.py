import datetime

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plotHistogramSubplots(df, param, name):
    # create a 3 x 3 subplot array for the months between Jan and Jul

    fig, ax = plt.subplots(3, 3, figsize=(20, 10))
    fig.suptitle(name)

    to_plot1 = df[df['Month'] == 1]
    to_plot_param1 = to_plot1[param]
    sns.distplot(np.array(to_plot_param1), kde=False, ax=ax[0, 0])
    # ax[0, 0].set_title('January 2020')
    ax[0, 0].set(xlabel='January 2020', ylabel='count', xlim=(-16, -12))

    to_plot2 = df[df['Month'] == 2]
    to_plot_param2 = to_plot2[param]
    sns.distplot(np.array(to_plot_param2), kde=False, ax=ax[0, 1])
    # ax[0, 1].set_title('February 2020')
    ax[0, 1].set(xlabel='February 2020', ylabel='count', xlim=(-16, -12))

    to_plot4 = df[df['Month'] == 4]
    to_plot_param4 = to_plot4[param]
    sns.distplot(np.array(to_plot_param4), kde=False, ax=ax[1, 0])
    # ax[1, 0].set_title('April 2020')
    ax[1, 0].set(xlabel='April 2020', ylabel='count', xlim=(-16, -12))

    to_plot5 = df[df['Month'] == 5]
    to_plot_param5 = to_plot5[param]
    sns.distplot(np.array(to_plot_param5), kde=False, ax=ax[1, 1])
    # ax[1, 1].set_title('May 2020')
    ax[1, 1].set(xlabel='May 2020', ylabel='count', xlim=(-16, -12))

    to_plot3 = df[df['Month'] == 3]
    to_plot_param3 = to_plot3[param]
    sns.distplot(np.array(to_plot_param3), kde=False, ax=ax[0, 2])
    # ax[0, 2].set_title('March 2020')
    ax[0, 2].set(xlabel='March 2020', ylabel='count', xlim=(-16, -12))

    # ax[2, 0].plot(x, -y, 'tab:red')
    # ax[2, 0].set_title('Axis [1,1]')

    to_plot6 = df[df['Month'] == 6]
    to_plot_param6 = to_plot6[param]
    sns.distplot(np.array(to_plot_param6), kde=False, ax=ax[1, 2])
    # ax[1, 2].set_title('June 2020')
    ax[1, 2].set(xlabel='June 2020', ylabel='count', xlim=(-16, -12))

    to_plot7 = df[df['Month'] == 7]
    to_plot_param7 = to_plot7[param]
    sns.distplot(np.array(to_plot_param7), kde=False, ax=ax[2, 0])
    # ax[2, 1].set_title('July 2020')
    ax[2, 0].set(xlabel='July 2020', ylabel='count', xlim=(-16, -12))

    to_plot8 = df[df['Month'] == 8]
    to_plot_param8 = to_plot8[param]
    sns.distplot(np.array(to_plot_param8), kde=False, ax=ax[2, 1])
    # ax[2, 1].set_title('July 2020')
    ax[2, 1].set(xlabel='August 2020', ylabel='count', xlim=(-16, -12))

    to_plot9 = df[df['Month'] == 9]
    to_plot_param9 = to_plot9[param]
    sns.distplot(np.array(to_plot_param9), kde=False, ax=ax[2, 2])
    # ax[2, 1].set_title('July 2020')
    ax[2, 2].set(xlabel='September 2020', ylabel='count', xlim=(-16, -12))

    # ax[2, 2].plot(x, -y, 'tab:red')
    # ax[2, 2].set_title('Axis [1,1]')


def plotBarWhiskerSubplots(df, param, name):
    # create a 3 x 3 subplot array for the months between Jan and Jul

    fig, ax = plt.subplots(3, 3, figsize=(20, 10))
    fig.suptitle("Meterological Parameter Measurements")

    # to_plot1 = df[param[1]]
    sns.boxplot(x="Month", y=param[0], data=df, ax=ax[0, 0], color='skyblue')
    # ax[0, 0].set_title('January 2020')
    ax[0, 0].set(xlabel="Month", ylabel=name[0])

    # to_plot2 = df[df['Month'] == 2]
    # to_plot_param2 = to_plot2[param]
    sns.boxplot(x="Month", y=param[1], data=df, ax=ax[0, 1], color='skyblue')
    # ax[0, 1].set_title('February 2020')
    ax[0, 1].set(xlabel="Month", ylabel=name[1])

    # to_plot4 = df[df['Month'] == 4]
    # to_plot_param4 = to_plot4[param]
    sns.boxplot(x="Month", y=param[3], data=df, ax=ax[1, 0], color='skyblue')
    # ax[1, 0].set_title('April 2020')
    ax[1, 0].set(xlabel="Month", ylabel=name[3])

    # to_plot5 = df[df['Month'] == 5]
    # to_plot_param5 = to_plot5[param]
    sns.boxplot(x="Month", y=param[4], data=df, ax=ax[1, 1], color='skyblue')
    # ax[1, 1].set_title('May 2020')
    ax[1, 1].set(xlabel="Month", ylabel=name[4])

    # to_plot3 = df[df['Month'] == 3]
    # to_plot_param3 = to_plot3[param]
    sns.boxplot(x="Month", y=param[2], data=df, ax=ax[0, 2], color='skyblue')
    # ax[0, 2].set_title('March 2020')
    ax[0, 2].set(xlabel="Month", ylabel=name[2])

    # to_plot7 = df[df['Month'] == 7]
    # to_plot_param7 = to_plot7[param]
    sns.boxplot(x="Month", y=param[6], data=df, ax=ax[2, 0], color='skyblue')
    # ax[2, 1].set_title('July 2020')
    ax[2, 0].set(xlabel="Month", ylabel=name[6])

    # to_plot6 = df[df['Month'] == 6]
    # to_plot_param6 = to_plot6[param]
    sns.boxplot(x="Month", y=param[5], data=df, ax=ax[1, 2], color='skyblue')
    # ax[1, 2].set_title('June 2020')
    ax[1, 2].set(xlabel="Month", ylabel=name[5])

    # to_plot8 = df[df['Month'] == 8]
    # to_plot_param8 = to_plot8[param]
    sns.boxplot(x="Month", y=param[8], data=df, ax=ax[2, 1], color='skyblue')
    # ax[2, 1].set_title('July 2020')
    ax[2, 1].set(xlabel="Month", ylabel=name[8])

    # to_plot9 = df[df['Month'] == 9]
    # to_plot_param9 = to_plot9[param]
    sns.boxplot(x="Month", y=param[7], data=df, ax=ax[2, 2], color='skyblue')
    # ax[2, 1].set_title('July 2020')
    ax[2, 2].set(xlabel="Month", ylabel=name[7])

def plotModelEvalKDE(df, mdl, mdl_name):
    g = sns.JointGrid(data=df, x=mdl, y="Log_Cn2", space=0)
    g.plot_joint(sns.kdeplot,
             fill=True, clip=((-16, -12), (-16, -12)),
             thresh=0, levels=100)
    #g.plot_marginals(sns.histplot, color="skyblue", alpha=1, bins=25)
    #g.plot_scatter(sns.scatter, color="skyblue", alpha=1, bins=25)


def plotDayMulti(df, idx, length, param, param2, param3, param4, param5, name2, name3, name4, name5):
    df_to_plot = df.iloc[idx:idx + length]
    dates = pd.to_datetime(df_to_plot['Timestamp'].values)

    # Make the plot
    f, ax = plt.subplots(2, 1)
    ax[0].set(yscale='log')
    df_to_plot.index = dates

    sns.regplot(y=param, x=df_to_plot.index.values, data=df_to_plot, fit_reg=False, label='Measured Cn2', marker='+',
                color='midnightblue', scatter_kws={'s': 5})
    sns.regplot(y=param2, x=df_to_plot.index.values, data=df_to_plot, fit_reg=False, label=name2, marker='o',
                color='firebrick', scatter_kws={'s': 2})
    sns.regplot(y=param3, x=df_to_plot.index.values, data=df_to_plot, fit_reg=False, label=name3, marker='o',
                color='darkorange', scatter_kws={'s': 2})
    sns.regplot(y=param4, x=df_to_plot.index.values, data=df_to_plot, fit_reg=False, label=name4, marker='o',
                color='darkorchid', scatter_kws={'s': 2})
    sns.regplot(y=param5, x=df_to_plot.index.values, data=df_to_plot, fit_reg=False, label=name5, marker='o',
                color='lightpink', scatter_kws={'s': 2})
    ax[0] = sns.regplot(y=param, x=df_to_plot.index.values, data=df_to_plot, fit_reg=False, marker='o',
                        color='midnightblue', scatter_kws={'s': 2})

    ax[0].set_ylim(1e-20, 1e-12)
    ax[0].set_xlim(dates[0], dates[-1])

    plt.setp(ax[0].get_xticklabels(), rotation=45)
    # plt.autofmt_xdate()

    ## rotate and align the tick labels so they look better
    # fig.autofmt_xdate()

    # use a more precise date string for the x axis locations in the
    # toolbar
    # ax.fmt_xdata = mdates.DateFormatter('%m-%d %H')

    ax[0].legend(prop={"size": 8})

    ax[0].set(yscale='log')
    ax[0].set(xlabel=dates[0].strftime("%Y/%m/%d"), ylabel='Observed Cn2')
    ax[0].fmt_xdata = mdates.DateFormatter('%H:%M')

    ax[1] = sns.regplot(y='Relative_Humidity', x=df_to_plot.index.values, data=df_to_plot, fit_reg=False, marker='o',
                        color='midnightblue', scatter_kws={'s': 2})
    ax[1].set_xlim(dates[0], dates[-1])

    plt.setp(ax[1].get_xticklabels(), rotation=45)
    # plt.autofmt_xdate()

    ## rotate and align the tick labels so they look better
    # fig.autofmt_xdate()

    # use a more precise date string for the x axis locations in the
    # toolbar
    # ax.fmt_xdata = mdates.DateFormatter('%m-%d %H')

    ax[1].legend(prop={"size": 8})
    ax[1].set(xlabel=dates[0].strftime("%Y/%m/%d"), ylabel='Air Temperature')
    ax[1].fmt_xdata = mdates.DateFormatter('%H:%M')

    # compare the predicted and measured Cn2
    mse = np.mean(np.square(df_to_plot[param] - df_to_plot[param2]))
    mae = np.mean((np.absolute(df_to_plot[param] - df_to_plot[param2])))
    mae2 = np.mean((np.absolute(df_to_plot[param] - df_to_plot[param3])))

    log_mse = np.mean(np.square(np.log10(df_to_plot[param]) - np.log10(df_to_plot[param2])))
    log_mae = np.mean((np.absolute(np.log10(df_to_plot[param]) - np.log10(df_to_plot[param2]))))
    # log_mae2 = np.mean((np.absolute(np.log10(df_to_plot[param]) - np.log10(df_to_plot[param3]))))

    print(mse)
    print(mae)
    print(mae2)
    print(log_mse)
    print(log_mae)
    plt.show()


def plotDayTimeseries(df, date, param):
    str_date = date.astype(str)
    next_date = date + np.timedelta64(24, 'h')
    mask = (df['Timestamp'] >= date) & (df['Timestamp'] <= next_date)
    df_to_plot = df.loc[mask]
    # Make the plot
    f, ax = plt.subplots()
    ax.set(yscale='log')
    sns.regplot(y=param, x=df_to_plot.index, data=df_to_plot, fit_reg=False, marker='o', color='midnightblue',
                scatter_kws={'s': 2})
    ax = sns.regplot(y=param, x=df_to_plot.index, data=df_to_plot, fit_reg=False, marker='o', color='midnightblue',
                     scatter_kws={'s': 2})
    ax.set_ylim(1e-17, 1e-13)
    ax.set(yscale='log')
    ax.set(xlabel=str_date, ylabel='Observed Cn2')
    # xlabel = str_date
    # ylabel = param


# Time series plot of one day
def plotDayTimeseriesPrediction(df, date, param, param2):
    str_date = date.astype(str)
    next_date = date + np.timedelta64(24, 'h')
    mask = (df['Timestamp'] >= date) & (df['Timestamp'] <= next_date)
    df_to_plot = df.loc[mask]
    # Make the plot
    f, ax = plt.subplots()
    ax.set(yscale='log')
    sns.regplot(y=param, x=df_to_plot.index, data=df_to_plot, fit_reg=False, marker='o', color='midnightblue',
                scatter_kws={'s': 2})
    sns.regplot(y=param2, x=df_to_plot.index, data=df_to_plot, fit_reg=False, marker='o', color='firebrick',
                scatter_kws={'s': 2})
    ax = sns.regplot(y=param, x=df_to_plot.index, data=df_to_plot, fit_reg=False, marker='o', color='midnightblue',
                     scatter_kws={'s': 2})
    ax.set_ylim(1e-17, 1e-13)
    ax.set(yscale='log')
    ax.set(xlabel=str_date, ylabel='Observed Cn2')
    # xlabel = str_date
    # ylabel = param


# Time series plot of one
def plotDayTimeseriesPredictionByIndexSingle(df, idx, length, param, param2):
    df_to_plot = df.iloc[idx:idx + length]
    # df_to_plot[param2] = 10 ** df_to_plot[param2]
    # Make the plot
    f, ax = plt.subplots()
    ax.set(yscale='log')
    # ax.set_xlim(idx,idx+length)

    # df_to_plot.index  = df_to_plot['Timestamp']

    sns.regplot(y=param, x=df_to_plot.index, data=df_to_plot, fit_reg=False, label='Measured Cn2', marker='o',
                color='midnightblue', scatter_kws={'s': 2})
    sns.regplot(y=param2, x=df_to_plot.index, data=df_to_plot, fit_reg=False, label='Macro-Met Cn2 Prediction',
                marker='o', color='firebrick', scatter_kws={'s': 2})
    # sns.regplot(y = param3, x = df_to_plot.index, data=df_to_plot, fit_reg=False, label='Offshore Model Cn2 Prediction', marker='o', color='darkorange', scatter_kws={'s':2})
    ax = sns.regplot(y=param, x=df_to_plot.index, data=df_to_plot, fit_reg=False, marker='o', color='midnightblue',
                     scatter_kws={'s': 2})
    ax.set_ylim(1e-17, 1e-12)
    ax.legend()

    ax.set(yscale='log')
    ax.set(xlabel='Timestamp', ylabel='Observed Cn2')

    # compare the predicted and measured Cn2
    mse = np.mean(np.square(df_to_plot[param] - df_to_plot[param2]))
    mae = np.mean((np.absolute(df_to_plot[param] - df_to_plot[param2])))
    # mae2 = np.mean((np.absolute(df_to_plot[param] - df_to_plot[param3])))

    log_mse = np.mean(np.square(np.log10(df_to_plot[param]) - np.log10(df_to_plot[param2])))
    log_mae = np.mean((np.absolute(np.log10(df_to_plot[param]) - np.log10(df_to_plot[param2]))))
    # log_mae2 = np.mean((np.absolute(np.log10(df_to_plot[param]) - np.log10(df_to_plot[param3]))))

    print(mse)
    print(mae)
    # print(mae2)
    print(log_mse)
    print(log_mae)
    plt.show()


# Time series plot of one day
def plotDayTimeseriesPredictionByIndexBackup(df, idx, length, param, param2, param3, name2, name3):
    df_to_plot = df.iloc[idx:idx + length]
    # Make the plot
    f, ax = plt.subplots()
    ax.set(yscale='log')
    # ax.set_xlim(idx,idx+length)

    # df_to_plot.index  = df_to_plot['Timestamp']

    sns.regplot(y=param, x=df_to_plot.index, data=df_to_plot, fit_reg=False, label='Measured Cn2', marker='o',
                color='midnightblue', scatter_kws={'s': 2})
    sns.regplot(y=param2, x=df_to_plot.index, data=df_to_plot, fit_reg=False, label=name2, marker='o',
                color='firebrick', scatter_kws={'s': 2})
    sns.regplot(y=param3, x=df_to_plot.index, data=df_to_plot, fit_reg=False, label=name3, marker='o',
                color='darkorange', scatter_kws={'s': 2})
    ax = sns.regplot(y=param, x=df_to_plot.index, data=df_to_plot, fit_reg=False, marker='o', color='midnightblue',
                     scatter_kws={'s': 2})
    ax.set_ylim(1e-17, 1e-12)
    ax.legend()

    ax.set(yscale='log')
    ax.set(xlabel='Timestamp', ylabel='Observed Cn2')

    # compare the predicted and measured Cn2
    mse = np.mean(np.square(df_to_plot[param] - df_to_plot[param2]))
    mae = np.mean((np.absolute(df_to_plot[param] - df_to_plot[param2])))
    mae2 = np.mean((np.absolute(df_to_plot[param] - df_to_plot[param3])))

    log_mse = np.mean(np.square(np.log10(df_to_plot[param]) - np.log10(df_to_plot[param2])))
    log_mae = np.mean((np.absolute(np.log10(df_to_plot[param]) - np.log10(df_to_plot[param2]))))
    # log_mae2 = np.mean((np.absolute(np.log10(df_to_plot[param]) - np.log10(df_to_plot[param3]))))

    print(mse)
    print(mae)
    print(mae2)
    print(log_mse)
    print(log_mae)
    plt.show()


# Time series plot of one day
def plotDayTimeseriesPredictionTestData(df, param, param2, param3):
    df_to_plot = df
    # Make the plot
    f, ax = plt.subplots()
    ax.set(yscale='log')
    # ax.set_xlim(idx,idx+length)

    # df_to_plot.index  = df_to_plot['Timestamp']

    sns.regplot(y=param, x=df_to_plot.index, data=df_to_plot, fit_reg=False, label='Measured Cn2', marker='o',
                color='midnightblue', scatter_kws={'s': 2})
    sns.regplot(y=param2, x=df_to_plot.index, data=df_to_plot, fit_reg=False, label='Random Forest Cn2 Prediction',
                marker='o', color='firebrick', scatter_kws={'s': 2})
    sns.regplot(y=param3, x=df_to_plot.index, data=df_to_plot, fit_reg=False, label='Boosted Model Cn2 Prediction',
                marker='o', color='darkorange', scatter_kws={'s': 2})
    ax = sns.regplot(y=param, x=df_to_plot.index, data=df_to_plot, fit_reg=False, marker='o', color='midnightblue',
                     scatter_kws={'s': 2})
    ax.set_ylim(1e-17, 1e-12)
    ax.legend()

    ax.set(yscale='log')
    ax.set(xlabel='Timestamp', ylabel='Observed Cn2')

    # compare the predicted and measured Cn2
    mse = np.mean(np.square(df_to_plot[param] - df_to_plot[param2]))
    mae = np.mean((np.absolute(df_to_plot[param] - df_to_plot[param2])))
    mae2 = np.mean((np.absolute(df_to_plot[param] - df_to_plot[param3])))

    log_mse = np.mean(np.square(np.log10(df_to_plot[param]) - np.log10(df_to_plot[param2])))
    log_mae = np.mean((np.absolute(np.log10(df_to_plot[param]) - np.log10(df_to_plot[param2]))))
    # log_mae2 = np.mean((np.absolute(np.log10(df_to_plot[param]) - np.log10(df_to_plot[param3]))))

    print(mse)
    print(mae)
    print(mae2)
    print(log_mse)
    print(log_mae)
    plt.show()


# Time series plot of one day
def plotDayTimeseriesPredictionTestDataLog(df, idx, length, param, param2):
    df_to_plot = df.iloc[idx:idx + length]
    df_to_plot = df
    # Make the plot
    f, ax = plt.subplots()
    # ax.set(yscale='log')
    # ax.set_xlim(idx,idx+length)

    # df_to_plot.index  = df_to_plot['Timestamp']

    sns.regplot(y=param, x=df_to_plot.index, data=df_to_plot, fit_reg=False, label='Measured Cn2', marker='o',
                color='midnightblue', scatter_kws={'s': 2})
    sns.regplot(y=param2, x=df_to_plot.index, data=df_to_plot, fit_reg=False, label='Random Forest Cn2 Prediction',
                marker='o', color='firebrick', scatter_kws={'s': 2})
    # sns.regplot(y = param3, x = df_to_plot.index, data=df_to_plot, fit_reg=False, label='Boosted Model Cn2 Prediction', marker='o', color='darkorange', scatter_kws={'s':2})
    ax = sns.regplot(y=param, x=df_to_plot.index, data=df_to_plot, fit_reg=False, marker='o', color='midnightblue',
                     scatter_kws={'s': 2})
    ax.set_ylim(-17, -12)
    ax.legend()

    # ax.set( yscale='log')
    ax.set(xlabel='Timestamp', ylabel='Observed Cn2')

    # compare the predicted and measured Cn2
    mse = np.mean(np.square(df_to_plot[param] - df_to_plot[param2]))
    mae = np.mean((np.absolute(df_to_plot[param] - df_to_plot[param2])))
    # mae2 = np.mean((np.absolute(df_to_plot[param] - df_to_plot[param3])))

    # log_mse = np.mean(np.square(np.log10(df_to_plot[param]) - np.log10(df_to_plot[param2])))
    # log_mae = np.mean((np.absolute(np.log10(df_to_plot[param]) - np.log10(df_to_plot[param2]))))
    # log_mae2 = np.mean((np.absolute(np.log10(df_to_plot[param]) - np.log10(df_to_plot[param3]))))

    print(mse)
    print(mae)
    # print(mae2)
    # print(log_mse)
    # print(log_mae)
    plt.show()


def plotPredictionTestData(df, param, param2, param3):
    df_to_plot = df
    # Make the plot
    f, ax = plt.subplots()
    ax.set(yscale='log')
    ax.set(xscale='log')
    # df_to_plot.index  = df_to_plot['Timestamp']

    sns.regplot(y=param2, x=param, data=df_to_plot, fit_reg=False, label='Random Forest Cn2 Prediction Accuracy',
                marker='o', color='midnightblue', scatter_kws={'s': 2})
    sns.regplot(y=param3, x=param, data=df_to_plot, fit_reg=False, label='Random Forest Cn2 Prediction', marker='o',
                color='firebrick', scatter_kws={'s': 2})
    ax = sns.regplot(y=param2, x=param, data=df_to_plot, fit_reg=False, marker='o', color='midnightblue',
                     scatter_kws={'s': 2})
    ax.set_ylim(1e-17, 1e-12)
    ax.set_xlim(1e-17, 1e-12)

    ax.legend()

    ax.set(yscale='log')
    ax.set(xscale='log')
    ax.set(xlabel='Measured Cn2', ylabel='Predicted Cn2')
    plt.show()


# Make monthly histograms
def plotMonthlyHistogram(df, param):
    for x in np.arange(df['Month'].value_counts().count()):
        month_num = df['Month'].unique()
        to_plot = df[df['Month'] == month_num[x]]
        to_plot_param = to_plot[param]
        # Make the plot
        plt.figure(x)
        sns.distplot(np.array(to_plot_param))
        xlabel = pd.Series(to_plot_param, name=param)
        sns.distplot(xlabel)


# Make monthly joint hex plots
def plotMonthlyJointHex(df, param1, param2):
    for x in np.arange(df['Month'].value_counts().count()):
        month_num = df['Month'].unique()
        to_plot = df[df['Month'] == month_num[x]]
        # Make the plot
        plt.figure(x)
        sns.jointplot(x=param1, y=param2, data=to_plot, kind='hex', color='blue')
        plt.title('Month = ' + str(month_num[x]))


# Make monthly joint hex plots
def plotMonthlyJointKDE(df, param1, param2):
    for x in np.arange(df['Month'].value_counts().count()):
        month_num = df['Month'].unique()
        to_plot = df[df['Month'] == month_num[x]]
        # Make the plot
        plt.figure(x)
        sns.jointplot(x=param1, y=param2, data=to_plot, kind='kde', color='blue')
        plt.title('Month = ' + str(month_num[x]))


# Make yearly joint hex plots
def plotYearlyJointHex(df, param1, param2):
    # Make the plot
    plt.figure()
    sns.jointplot(x=param1, y=param2, data=df, kind='hex', color='blue')


# Make yearly joint hex plots
def plotYearlyJointKDE(df, param1, param2):
    # Make the plot
    plt.figure()
    sns.jointplot(x=param1, y=param2, data=df, kind='kde', color='blue')


# Make monthly bar and whisker plots
def plotTotalBarWhisker(df, param):
    to_plot_param = df
    # Make the plot
    plt.figure()
    sns.boxplot(x='Month', y=param, data=to_plot_param)
    xlabel = pd.Series(to_plot_param, name=param)
    sns.boxplot(xlabel)


# Make violin Plot
def plotTotalViolin(df, param):
    to_plot_param = df
    # Make the plot
    plt.figure()
    sns.violinplot(x='Month', y=param, data=to_plot_param, inner='quartile')
    # xlabel = pd.Series(to_plot_param, name=param)
    # sns.boxplot(xlabel)


# Make monthly swarm plots
def plotTotalSwarm(df, param):
    to_plot_param = df
    # Make the plot
    plt.figure()
    # sns.boxplot(x='Month',y=param, data=to_plot_param)
    # xlabel = pd.Series(to_plot_param, name=param)
    sns.swarmplot(x='Month', y=param, data=to_plot_param, color='0.28')
    # sns.swarmplot(xlabel)


def plotYearlyHistogram(df, param):
    to_plot_param = df[param]
    # Make the plot
    plt.figure()
    sns.distplot(np.array(to_plot_param), kde=False, hist=False, fit=stats.norm)
    (mu, sigma) = stats.norm.fit(np.array(to_plot_param))
    plt.legend(["normal dist. fit ($\mu=${0:.2g}, $\sigma=${1:.2f})".format(mu, sigma)], loc=1)
    xlabel = pd.Series(to_plot_param, name=param)
    sns.distplot(xlabel)

