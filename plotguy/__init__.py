import os
import itertools
import multiprocessing as mp
import zlib
import requests
from bs4 import BeautifulSoup
import pandas as pd
import polars as pl
import numpy as np

from .equity_curves import *
from .signals import *
from .aggregate import *


def multi_process(function, parameters, number_of_core=8):
    pool = mp.Pool(processes=number_of_core)
    pool.map(function, parameters)
    pool.close()


def filename_only(para_combination):
    para_dict = para_combination['para_dict']
    start_date = para_combination['start_date']
    end_date = para_combination['end_date']
    py_filename = para_combination['py_filename']
    start_date_str = datetime.datetime.strptime(start_date, '%Y-%m-%d').strftime("%Y%m%d")
    end_date_str = datetime.datetime.strptime(end_date, '%Y-%m-%d').strftime("%Y%m%d")
    summary_mode = para_combination['summary_mode']
    freq = para_combination['freq']

    save_name = f'file={py_filename}&date={start_date_str}{end_date_str}&freq={freq}&summary_mode={summary_mode}&'

    for key in list(para_dict.keys()):
        para = para_combination[key]
        if key == 'code':
            if str(para).isdigit():
                para = str(para).zfill(5)

        if isinstance(para, float):
            if para.is_integer():
                para = int(para)

        save_name += f'{key}={str(para)}&'

    return save_name


def path_reference_code(save_name):
    reference_code = str(zlib.crc32(bytes(save_name, 'UTF-8')))[:8]
    return reference_code


def generate_filepath(para_combination, folder=''):
    file_format = para_combination['file_format']

    if not file_format == 'parquet':
        file_format = 'csv'

    save_name = filename_only(para_combination)
    reference_code = path_reference_code(save_name)
    output_folder = para_combination['output_folder']
    filepath = os.path.join(folder, output_folder, f'{para_combination["code"]}_{reference_code}.{file_format}')

    return filepath


def apply_pnl(row):
    # This is one line of backtest, i.e. one trade
    # Purpose if this is to calculate unrealized pnl between open date and close date, aggreagte same day pnl with other trade,
    # also save realized pnl

    df_daily = apply_pnl.df_daily

    open_date = row.open_date
    close_date = row.date
    now_close = float(df_daily.at[open_date, 'close'])
    dates = pd.date_range(start=open_date, end=close_date)
    open_price = float(row.open_price)
    num_of_share = int(row.num_of_share)
    commission = float(row.commission)
    action = row.action
    realized_pnl = float(row.realized_pnl)

    last_realized_capital = apply_pnl.last_realized_capital
    multiplier = float(apply_pnl.multiplier)


    for date in dates:
        if (date == open_date or date == close_date):
            # open_date and close_date may be the first
            # therefore, process open_date first, then close_date

            if date == open_date:
                # Unrealized pnl on the open date, record on df_daily
                df_daily.at[open_date, 'action'] = 'open'
                df_daily.at[open_date, 'signal_value'] = df_daily.at[open_date, 'bah']  # Mark the open position for analysi chart

                unrealized_pnl = num_of_share * multiplier * (now_close - open_price)#  - commission
                unrealized_pnl = round(unrealized_pnl, 3)
                df_daily.at[open_date, 'unrealized_pnl'] = unrealized_pnl

                # print(open_date, 'Open!', open_price, now_close, num_of_share, unrealized_pnl)

            if date == close_date:  # close position on this date, end of this trade
                # this is a same day trade
                # clear the open unrealized pnl as open and close same day should not aggregate
                # it should be record as realized pnl directly
                if open_date == close_date:
                    df_daily.at[date, 'unrealized_pnl'] = None

                # unrealized_pnl
                if df_daily.at[date, 'unrealized_pnl']:  # if there is already unrealized_pnl, aggregate
                    df_daily.at[date, 'unrealized_pnl'] = df_daily.at[date, 'unrealized_pnl'] + realized_pnl
                else:
                    df_daily.at[date, 'unrealized_pnl'] = realized_pnl

                # realized pnl
                if df_daily.at[date, 'realized_pnl']:  # if there is already realized_pnl, aggregate
                    df_daily.at[date, 'realized_pnl'] = df_daily.at[date, 'realized_pnl'] + realized_pnl
                else:
                    df_daily.at[date, 'realized_pnl'] = realized_pnl

                df_daily.at[date, 'action'] = action
                df_daily.at[date, 'commission'] = commission
                last_realized_capital = last_realized_capital + realized_pnl

                # print(date, 'Close!', open_price, close_date, realized_pnl)
                # print()

        else:
            # it is not (date == open_date or date == close_date), so it is between two date
            # calculate unrealized pnl only
            try:
                now_close = df_daily.at[date, 'close']

                unrealized_pnl = num_of_share * multiplier * (now_close - open_price) - commission
                unrealized_pnl = round(unrealized_pnl, 3)
                if df_daily.at[date, 'unrealized_pnl']:
                    df_daily.at[date, 'unrealized_pnl'] = df_daily.at[date, 'unrealized_pnl'] + unrealized_pnl
                else:
                    df_daily.at[date, 'unrealized_pnl'] = unrealized_pnl

                # print(date, 'Not yet close', open_price, now_close, unrealized_pnl)

            except:
                pass
                # print(date, 'Not yet close', 'Holiday!')

    apply_pnl.df_daily = df_daily


def df_daily_equity(row):
    last_equity_value = df_daily_equity.last_equity_value
    if row.name == 0:
        return last_equity_value
    else:
        if not (row.realized_pnl == None):
            equity_value = last_equity_value + row.realized_pnl
            df_daily_equity.last_equity_value = equity_value
        elif not (row.unrealized_pnl == None):
            equity_value = last_equity_value + row.unrealized_pnl
        else:
            equity_value = None

        return equity_value


def mp_cal_performance(tuple_data):
    para_combination = tuple_data[0]
    manager_list = tuple_data[1]

    result = cal_performance(para_combination)

    # new
    para_df_dict = {}

    para_df_dict['reference_code'] = path_reference_code(filename_only(para_combination))
    para_df_dict['reference_index'] = para_combination['reference_index']

    keys_to_keep = para_combination['para_dict'].keys()
    para_df_dict.update({k: v for k, v in para_combination.items() if k in keys_to_keep})
    para_df_dict.update(result)
    manager_list.append(para_df_dict)


def reference_code_apply(row):
    return path_reference_code(filename_only(reference_code_apply.all_para_combination[row.reference_index]))


def generate_backtest_result(all_para_combination, number_of_core=8, risk_free_rate='geometric_mean'):
    ## Get / Calculate risk free rate
    start_date = all_para_combination[0]['start_date']
    end_date = all_para_combination[0]['end_date']

    if isinstance(risk_free_rate, str):
        try:
            if risk_free_rate == 'geometric_mean':
                start_date_year = datetime.datetime.strptime(start_date, '%Y-%m-%d').year
                end_date_year = datetime.datetime.strptime(end_date, '%Y-%m-%d').year
                risk_free_rate = plotguy.get_geometric_mean_of_yearly_rate(start_date_year, end_date_year)
            else:
                risk_free_rate = plotguy.get_latest_fed_fund_rate()
        except:
            risk_free_rate = 2  # if network error, set rate to 2 %
            print('Network error. Risk free rate: {:.2f} %'.format(risk_free_rate))
    else:
        print('Risk free rate: {:.2f} %'.format(risk_free_rate))

    print('Backtest result is loading. Please wait patiently.')

    manager_list = mp.Manager().list()  # To save the result with index number

    cal_performance_list = []
    for para_combination in all_para_combination:
        para_combination['risk_free_rate'] = risk_free_rate
        cal_performance_list.append((para_combination, manager_list))

    pool = mp.Pool(processes=number_of_core)
    pool.map(mp_cal_performance, cal_performance_list)
    pool.close()

    df_backtest_result = pd.DataFrame(list(manager_list))
    df_backtest_result = df_backtest_result.sort_values(by='reference_index')
    df_backtest_result.reset_index(drop=True)
    df_backtest_result.to_csv('backtest_result.csv')


def plot_signal_analysis(py_filename, output_folder, start_date, end_date, para_dict, signal_settings):
    app = signals.Signals(py_filename, output_folder, start_date, end_date, para_dict, generate_filepath,
                          signal_settings)

    return app


def plot(mode, all_para_combination={}, subchart_settings={}, number_of_curves=20, risk_free_rate='geometric_mean'):
    if subchart_settings == {}:  # subchart default setting
        subchart_settings = {
            'histogram_period': [1, 3, 5, 10, 20],
            'subchart_1': ['volume', 'line']
        }

    if mode == 'equity_curves':
        result_df = pd.read_csv('backtest_result.csv', index_col=0)
        app = equity_curves.Plot(all_para_combination, result_df, subchart_settings, number_of_curves)

    if mode == 'aggregate':
        app = aggregate.Aggregate(risk_free_rate)

    if mode == 'signal_analysis':
        app = signals.Signals( all_para_combination, generate_filepath, subchart_settings)

    return app


def get_latest_fed_fund_rate():
    url = "https://fred.stlouisfed.org/series/FEDFUNDS"
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")

    fed_funds_rate = soup.find("span", class_="series-meta-observation-value").text
    print("Latest Federal Funds Rate:", fed_funds_rate, '%')
    # fed_funds_rate = float(fed_funds_rate) / 100
    fed_funds_rate = round(float(fed_funds_rate), 2)
    return fed_funds_rate


def get_geometric_mean_of_yearly_rate(start_year, end_year):  # backtest period
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"
    response = requests.get(url)
    data = response.text.split("\n")[:-1]
    data = [row.split(",") for row in data]
    df = pd.DataFrame(data[1:], columns=data[0])
    df.columns = ["date", "risk_free_rate"]
    df["date"] = pd.to_datetime(df["date"])
    df["risk_free_rate"] = pd.to_numeric(df["risk_free_rate"], errors='coerce')
    df.dropna(subset=['risk_free_rate'], inplace=True)

    risk_free_rate_history_yearly = df.resample("A", on="date").mean()
    risk_free_rate_history_yearly = risk_free_rate_history_yearly.round(3)

    # show only start between start_year and end_year
    risk_free_rate_history_yearly = risk_free_rate_history_yearly[
        risk_free_rate_history_yearly.index.year >= start_year]
    risk_free_rate_history_yearly = risk_free_rate_history_yearly[risk_free_rate_history_yearly.index.year <= end_year]

    fed_fund_rate_geometric_mean = np.exp(np.log(risk_free_rate_history_yearly["risk_free_rate"]).mean())
    fed_fund_rate_geometric_mean = round(fed_fund_rate_geometric_mean, 2)
    print("Federal Funds Rate Geometric mean from {} to {}: {} %".format(start_year, end_year,
                                                                         fed_fund_rate_geometric_mean))

    return fed_fund_rate_geometric_mean


def calculate_mdd(df, col):
    roll_max = df[col].cummax()
    daily_drawdown = df[col] / roll_max - 1.0
    max_daily_drawdown = daily_drawdown.cummin()

    return min(list(max_daily_drawdown)), min(list(df[col] - roll_max))


def calculate_win_rate_info(df):
    num_of_trade = list(df['action'] == 'open').count(True)
    num_of_loss = list(df['pnl'] < 0).count(True)
    num_of_win = num_of_trade - num_of_loss

    if num_of_trade > 0:
        win_rate = round(100 * num_of_win / num_of_trade, 2)
        loss_rate = round(100 * num_of_loss / num_of_trade, 2)
    else:
        win_rate = '--'
        loss_rate = '--'

    return num_of_trade, num_of_loss, num_of_win, win_rate, loss_rate


def calculate_win_rate(df_csv):
    df = df_csv[['date', 'realized_pnl', 'action']].copy()
    df = df[df['action'].notnull()].reset_index(drop=True)
    df = df.loc[df['action'] != '']  # for parquet
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['pnl'] = df['realized_pnl'].shift(-1)
    df['year'] = pd.DatetimeIndex(df['date']).year
    year_list = list(set(df['year']))
    year_list.sort()

    win_rate_dict = {}
    win_rate_dict['Overall'] = calculate_win_rate_info(df)

    for year in year_list:
        win_rate_dict[year] = calculate_win_rate_info(df.loc[df['year'] == year])

    return win_rate_dict


def calculate_sharpe_ratio(df, col, risk_free_rate):
    holding_period_day = (df.loc[df.index[-1], 'date'] - df.loc[df.index[0], 'date']).days
    net_profit = df.at[df.index[-1], col] - df.at[df.index[0], col]

    initial_capital = df.loc[df.index[0], col]

    # To avoid power error below
    if net_profit < 0 and abs(net_profit) > initial_capital:
        net_profit = initial_capital * -1

    equity_value_pct_series = df[col].pct_change()
    equity_value_pct_series = equity_value_pct_series.dropna()

    return_on_capital = net_profit / initial_capital
    annualized_return = (np.sign(1 + return_on_capital) * np.abs(1 + return_on_capital)) ** (
                365 / holding_period_day) - 1
    annualized_std = equity_value_pct_series.std() * math.sqrt(365)

    if annualized_std > 0:
        annualized_sr = (annualized_return - float(risk_free_rate) / 100) / annualized_std
    else:
        annualized_sr = 0

    return_on_capital = round(100 * return_on_capital, 2)
    annualized_return = round(100 * annualized_return, 2)
    annualized_std = round(100 * annualized_std, 2)
    annualized_sr = round(annualized_sr, 2)

    return net_profit, holding_period_day, return_on_capital, annualized_return, annualized_std, annualized_sr


def resample_summary_to_daily(para_combination, folder=''):
    # Need to deal with the start date and date

    start_date = para_combination['start_date']
    end_date = para_combination['end_date']
    start_date_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date_datetime = datetime.datetime.strptime(end_date, '%Y-%m-%d')

    sec_profile = para_combination['sec_profile']

    sectype = sec_profile['sectype']
    lot_size_dict = sec_profile['lot_size_dict']
    code = para_combination['code']
    lot_size = lot_size_dict[code]

    intraday = para_combination['intraday']
    freq = para_combination['freq']
    file_format = para_combination['file_format']
    sec_profile = para_combination['sec_profile']

    if sectype == 'FUT':
        for key in sec_profile:
            if 'margin_req' in key:
                sec_profile['margin_req'] = sec_profile.pop(key)
                break

        margin_req = sec_profile['margin_req']
        multiplier = sec_profile['multiplier']

    elif sectype == 'STK':
        multiplier = 1

    # Read backtest data
    save_path = generate_filepath(para_combination=para_combination, folder=folder)

    if file_format == 'parquet':
        df = pl.read_parquet(save_path)
    else:
        df = pl.read_csv(save_path, try_parse_dates=True)

    # Calculate initial capital before trim the dataframe
    if len(df) == 0:
        # This would happen when it is summary mode and no trade for the strategy
        # Note that in intraday but not summary mode, length of df of  no trade df_bakctest is still > 0
        initial_capital = sec_profile['initial_capital']
    else:
        if pl.sum(df.get_column('realized_pnl')) == None:  # when NOT summary mode but no trade
            initial_capital = sec_profile['initial_capital']
        else:
            initial_capital = df.row(-1, named=True)['equity_value'] - pl.sum(df.get_column('realized_pnl'))

    last_realized_capital = initial_capital

    df = (df.lazy()
          .filter((pl.col('action') != '') &
                  (pl.col('action').is_not_null()) &
                  (pl.col('date') >= start_date_datetime) &
                  (pl.col('date') <= end_date_datetime)
                  )
          .select(['date', 'action', 'open_price', 'close', 'commission', 'num_of_share', 'realized_pnl'])
          .with_columns(pl.col('date').shift(1).alias('open_date'))
          .with_columns(pl.col('open_price').shift(1).alias('open_price'))
          .with_columns(pl.col('num_of_share').shift(1).alias('num_of_share'))
          .filter(pl.col('action') != 'open')
          .collect()
          )

    df_backtest = df.to_pandas()

    # Read source data
    data_folder = para_combination['data_folder']
    code = para_combination['code']

    data_path = os.path.join(folder, data_folder, f'{code}_{freq}.{file_format}')

    if file_format == 'parquet':
        df = pl.read_parquet(data_path)
    else:
        df = pl.read_csv(data_path, try_parse_dates=True)

    df = (df.lazy()
          .select(['datetime', 'open', 'high', 'low', 'close', 'volume']).sort("datetime")
          .sort("datetime")
          .groupby_dynamic("datetime", every="1d").agg([
        pl.col("open").first(),
        pl.col("high").max(),
        pl.col("low").min(),
        pl.col("close").last(),
        pl.col("volume").sum(),
    ])  # ploars automatically filter out non-trading date (null close dates in the df_csv)
          .filter((pl.col('datetime') >= start_date_datetime) & (pl.col('datetime') <= end_date_datetime))
          .with_columns(pl.col('datetime').alias('date'))
          .with_columns([
        (pl.col('close') * (initial_capital / df.head(1)['close'][0])).alias('bah'),
        pl.lit(None).alias('equity_value'),
        pl.lit(None).alias('action'),
        pl.lit(None).alias('realized_pnl'),
        pl.lit(None).alias('unrealized_pnl'),
        pl.lit(None).alias('signal_value'),   # for the position of analysis chart
        pl.lit(None).alias('commission'),
    ])
          .collect()
          )

    df_daily = df.to_pandas()
    df_daily.index = pd.to_datetime(df_daily['datetime'], format='%Y-%m-%d')

    # Apply apply_pnl on backtest here
    apply_pnl.last_realized_capital = last_realized_capital
    apply_pnl.multiplier = multiplier
    apply_pnl.df_daily = df_daily

    df_backtest.apply(apply_pnl, axis=1)

    df_daily = apply_pnl.df_daily

    df_daily = df_daily.reset_index(drop=True)
    df_daily_equity.last_equity_value = initial_capital
    df_daily['equity_value'] = df_daily.apply(df_daily_equity, axis=1).fillna(method='ffill')

    return df_daily


def cal_performance(para_combination):
    start_date = para_combination['start_date']
    end_date = para_combination['end_date']

    risk_free_rate = para_combination['risk_free_rate']

    intraday = para_combination['intraday']
    summary_mode = para_combination['summary_mode']

    file_format = para_combination['file_format']

    if (intraday or summary_mode):
        df_daily = resample_summary_to_daily(para_combination=para_combination)
    else:
        save_path = generate_filepath(para_combination=para_combination)
        if file_format == 'parquet':
            df_backtest = pd.read_parquet(save_path)  # Daraframe that may not be daily
        else:
            df_backtest = pd.read_csv(save_path, index_col=0)  # Daraframe that may not be daily

        df_backtest['date'] = pd.to_datetime(df_backtest['date'], format='%Y-%m-%d')
        df_backtest = df_backtest.loc[(df_backtest['date'] >= start_date) & (df_backtest['date'] <= end_date)]
        df_backtest = df_backtest.reset_index(drop=True)  # Reset Index
        df_daily = df_backtest

    df = df_daily


    # Deter if equity_value unchange = no tade
    equity_value_column = df['equity_value'].to_numpy()
    no_trade = (equity_value_column[0] == equity_value_column).all()

    result_dict = {}

    # Determine years at the beginning
    start_date_year = datetime.datetime.strptime(start_date, '%Y-%m-%d').year
    end_date_year = datetime.datetime.strptime(end_date, '%Y-%m-%d').year
    year_list = list(range(start_date_year, end_date_year + 1))
    for y in year_list: result_dict[str(y)] = []

    # if length of backtest is zero, no trade, no performance
    if no_trade:
        return_on_capital = 0
        result_dict['holding_period_day'] = 0
        result_dict['total_commission'] = 0
        result_dict['net_profit'] = 0
        result_dict['return_on_capital'] = 0
        result_dict['annualized_return'] = 0
        result_dict['annualized_std'] = 0
        result_dict['annualized_sr'] = 0
        result_dict['mdd_dollar'] = 0
        result_dict['mdd_pct'] = 0
        result_dict['num_of_trade'] = 0
        result_dict['win_rate'] = 0
        result_dict['loss_rate'] = 0
        result_dict['net_profit_to_mdd'] = np.inf
        result_dict['cov_count'] = 0
        result_dict['cov_return'] = 0

        # Win rate by year
        for year in year_list:
            result_dict[str(year)] = 0
            result_dict[f'{year}_win_rate'] = '--'
            result_dict[f'{year}_return'] = 0

    else:
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df['year'] = pd.DatetimeIndex(df['date']).year

        # Strategy Performance
        net_profit, holding_period_day, return_on_capital, annualized_return, annualized_std, annualized_sr = calculate_sharpe_ratio(
            df, 'equity_value', risk_free_rate)
        mdd_pct, mdd_dollar = calculate_mdd(df, 'equity_value')
        mdd_pct = mdd_pct * -100
        mdd_dollar = mdd_dollar * -1

        # Win Rate (need to use df_backtest directly)

        save_path = generate_filepath(para_combination=para_combination)
        if file_format == 'parquet':
            df_backtest = pd.read_parquet(save_path)  # Daraframe that may not be daily
        else:
            df_backtest = pd.read_csv(save_path, index_col=0)  # Daraframe that may not be daily

        win_rate_dict = calculate_win_rate(df_backtest)
        num_of_trade, num_of_loss, num_of_win, win_rate, loss_rate = win_rate_dict['Overall']
        # print('net_profit',net_profit)

        total_commission = df['commission'].sum()

        result_dict['holding_period_day'] = holding_period_day
        result_dict['total_commission'] = total_commission

        result_dict['net_profit'] = net_profit
        result_dict['return_on_capital'] = return_on_capital
        result_dict['annualized_return'] = annualized_return
        result_dict['annualized_std'] = annualized_std
        result_dict['annualized_sr'] = annualized_sr
        result_dict['mdd_dollar'] = mdd_dollar
        result_dict['mdd_pct'] = mdd_pct
        result_dict['num_of_trade'] = num_of_trade
        result_dict['win_rate'] = win_rate
        result_dict['loss_rate'] = loss_rate

        if mdd_dollar == 0:
            result_dict['net_profit_to_mdd'] = np.inf
        else:
            result_dict['net_profit_to_mdd'] = net_profit / mdd_dollar


        # Count and Win rate by year
        year_count = []
        for year in year_list:
            try:
                result_dict[str(year)] = win_rate_dict[year][0]
                result_dict[f'{year}_win_rate'] = win_rate_dict[year][3]

            except Exception as e:
                # print(e)
                result_dict[str(year)] = 0
                result_dict[f'{year}_win_rate'] = '--'

            year_count.append(result_dict[str(year)])


        # result_dict['cov_count'] = round(np.std(year_count) / np.mean(year_count), 3)

        std_dev = np.std(year_count)
        mean = np.mean(year_count)

        if std_dev != 0 and not np.isnan(std_dev) and mean != 0 and not np.isnan(mean):
            result_dict['cov_count'] = round(std_dev / mean, 3)
        else:
            result_dict['cov_count'] = 0  # Assign a value indicating an invalid result


        # Performance by year
        first_equity_value = 0
        last_equity_value = 0
        year_return_list = []
        for year in year_list:
            if not df.loc[df['year'] == year].empty:  # if trade
                if first_equity_value == 0:  # if 1st year, set beginning as the first equity_value
                    first_equity_value = df.loc[df['year'] == year].iloc[0].equity_value
                last_equity_value = df.loc[df['year'] == year].iloc[-1].equity_value
                yearly_return = (last_equity_value - first_equity_value) / first_equity_value
                if np.isnan(yearly_return):
                    result_dict[f'{year}_return'] = 0
                    year_return_list.append(0)
                else:
                    result_dict[f'{year}_return'] = int(yearly_return * 100)
                    year_return_list.append(int(yearly_return * 100))

            else:  # no trade
                result_dict[f'{year}_return'] = '-----'
                year_return_list.append(0)

            first_equity_value = last_equity_value

        # cov_return
        return_year_std = np.std(year_return_list)
        return_year_mean = np.mean(year_return_list)
        if return_year_mean == 0:
            cov_return = 0
        else:
            cov_return = round(return_year_std / return_year_mean, 3)
        result_dict['cov_return'] = cov_return

    result_dict['risk_free_rate'] = risk_free_rate




    # BaH Performance
    bah_net_return, holding_period_day, bah_return, \
    bah_annualized_return, bah_annualized_std, bah_annualized_sr = calculate_sharpe_ratio(df, 'close',
                                                                                         risk_free_rate)
    initial_capital = df.loc[df.index[0], 'equity_value']
    df['bah_equity_curve'] = df['close'] * initial_capital // df.loc[df.index[0], 'close']
    bah_mdd_pct, bah_mdd_dollar = calculate_mdd(df, 'bah_equity_curve')
    bah_mdd_pct = bah_mdd_pct * -100
    bah_mdd_dollar = bah_mdd_dollar * -1

    result_dict['bah_return'] = bah_return
    result_dict['bah_annualized_return'] = bah_annualized_return
    result_dict['bah_annualized_std'] = bah_annualized_std
    result_dict['bah_annualized_sr'] = bah_annualized_sr
    result_dict['bah_mdd_dollar'] = bah_mdd_dollar
    result_dict['bah_mdd_pct'] = bah_mdd_pct
    result_dict['return_to_bah'] = return_on_capital - bah_return

    return result_dict

