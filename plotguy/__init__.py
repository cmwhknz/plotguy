import datetime
import itertools
import math
import multiprocessing as mp
import os
import zlib

import numpy as np
import pandas as pd
import polars as pl
import requests
from bs4 import BeautifulSoup

from .aggregate import *
from .equity_curves import *
from .signals import *


def multi_process(function, parameters, number_of_cores: int = 8) -> None:
    """Run `function` over `parameters` in a process pool."""
    with mp.Pool(processes=number_of_cores) as pool:
        pool.map(function, parameters)


def filename_only(para_combination: dict) -> str:
    para_dict = para_combination["para_dict"]
    start_date = para_combination["start_date"]
    end_date = para_combination["end_date"]
    py_filename = para_combination["py_filename"]
    summary_mode = para_combination["summary_mode"]
    freq = para_combination["freq"]

    start_date_str = datetime.datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m%d")
    end_date_str = datetime.datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y%m%d")

    save_name = (
        f"file={py_filename}&date={start_date_str}{end_date_str}"
        f"&freq={freq}&summary_mode={summary_mode}&"
    )

    for key in list(para_dict.keys()):
        para = para_combination[key]

        if key == "code" and str(para).isdigit():
            para = str(para).zfill(5)

        if isinstance(para, float) and para.is_integer():
            para = int(para)

        save_name += f"{key}={para}&"

    return save_name


def path_reference_code(save_name: str) -> str:
    return str(zlib.crc32(bytes(save_name, "utf-8")))[:8]


def generate_filepath(para_combination: dict, folder: str = "") -> str:
    file_format = para_combination.get("file_format", "csv")
    file_format = "parquet" if file_format == "parquet" else "csv"

    save_name = filename_only(para_combination)
    reference_code = path_reference_code(save_name)
    output_folder = para_combination["output_folder"]

    return os.path.join(
        folder,
        output_folder,
        f'{para_combination["code"]}_{reference_code}.{file_format}',
    )


def apply_pnl(row) -> None:
    """
    Process one trade row to update daily unrealized/realized PnL in apply_pnl.df_daily.
    Uses function attributes:
      - apply_pnl.df_daily
      - apply_pnl.last_realized_capital
      - apply_pnl.multiplier
    """
    df_daily = apply_pnl.df_daily

    open_date = row.open_date
    close_date = row.date
    now_close = float(df_daily.at[open_date, "close"])

    dates = pd.date_range(start=open_date, end=close_date)
    open_price = float(row.open_price)
    num_of_share = int(row.num_of_share)
    commission = float(row.commission)
    action = row.action
    realized_pnl = float(row.realized_pnl)

    last_realized_capital = apply_pnl.last_realized_capital
    multiplier = float(apply_pnl.multiplier)

    for date in dates:
        if date in (open_date, close_date):
            if date == open_date:
                df_daily.at[open_date, "action"] = "open"
                # mark open position for analysis chart
                df_daily.at[open_date, "signal_value"] = df_daily.at[open_date, "bah"]

                unrealized_pnl = num_of_share * multiplier * (now_close - open_price)
                df_daily.at[open_date, "unrealized_pnl"] = round(unrealized_pnl, 3)

            if date == close_date:
                # same-day trade: don't keep unrealized on that day; realized goes straight in
                if open_date == close_date:
                    df_daily.at[date, "unrealized_pnl"] = None

                # aggregate unrealized_pnl on close date with realized_pnl
                if pd.notna(df_daily.at[date, "unrealized_pnl"]):
                    df_daily.at[date, "unrealized_pnl"] = df_daily.at[date, "unrealized_pnl"] + realized_pnl
                else:
                    df_daily.at[date, "unrealized_pnl"] = realized_pnl

                # aggregate realized_pnl
                if pd.notna(df_daily.at[date, "realized_pnl"]):
                    df_daily.at[date, "realized_pnl"] = df_daily.at[date, "realized_pnl"] + realized_pnl
                else:
                    df_daily.at[date, "realized_pnl"] = realized_pnl

                df_daily.at[date, "action"] = action
                df_daily.at[date, "commission"] = commission
                last_realized_capital = last_realized_capital + realized_pnl
        else:
            # between open and close: unrealized only
            try:
                now_close = df_daily.at[date, "close"]
                unrealized_pnl = num_of_share * multiplier * (now_close - open_price) - commission
                unrealized_pnl = round(unrealized_pnl, 3)

                if pd.notna(df_daily.at[date, "unrealized_pnl"]):
                    df_daily.at[date, "unrealized_pnl"] = df_daily.at[date, "unrealized_pnl"] + unrealized_pnl
                else:
                    df_daily.at[date, "unrealized_pnl"] = unrealized_pnl
            except Exception:
                # holiday / missing date
                pass

    apply_pnl.df_daily = df_daily
    apply_pnl.last_realized_capital = last_realized_capital


def df_daily_equity(row):
    last_equity_value = df_daily_equity.last_equity_value

    if row.name == 0:
        return last_equity_value

    if row.realized_pnl is not None:
        equity_value = last_equity_value + row.realized_pnl
        df_daily_equity.last_equity_value = equity_value
        return equity_value

    if row.unrealized_pnl is not None:
        return last_equity_value + row.unrealized_pnl

    return None


def mp_cal_performance(tuple_data) -> None:
    para_combination, manager_list = tuple_data
    result = cal_performance(para_combination)

    para_df_dict = {
        "reference_code": path_reference_code(filename_only(para_combination)),
        "reference_index": para_combination["reference_index"],
    }

    keys_to_keep = para_combination["para_dict"].keys()
    para_df_dict.update({k: v for k, v in para_combination.items() if k in keys_to_keep})
    para_df_dict.update(result)

    manager_list.append(para_df_dict)


def reference_code_apply(row):
    all_para = reference_code_apply.all_para_combination[row.reference_index]
    return path_reference_code(filename_only(all_para))


def generate_backtest_result(all_para_combination, number_of_cores: int = 8, risk_free_rate="geometric_mean") -> None:
    start_date = all_para_combination[0]["start_date"]
    end_date = all_para_combination[0]["end_date"]

    if isinstance(risk_free_rate, str):
        try:
            if risk_free_rate == "geometric_mean":
                start_year = datetime.datetime.strptime(start_date, "%Y-%m-%d").year
                end_year = datetime.datetime.strptime(end_date, "%Y-%m-%d").year
                risk_free_rate = get_geometric_mean_of_yearly_rate(start_year, end_year)
            else:
                risk_free_rate = get_latest_fed_fund_rate()
        except Exception:
            risk_free_rate = 2.0
            print(f"Network error. Risk free rate: {risk_free_rate:.2f} %")
    else:
        print(f"Risk free rate: {float(risk_free_rate):.2f} %")

    print("Backtest result is loading. Please wait patiently.")

    manager_list = mp.Manager().list()
    cal_performance_list = []
    for para_combination in all_para_combination:
        para_combination["risk_free_rate"] = risk_free_rate
        cal_performance_list.append((para_combination, manager_list))

    with mp.Pool(processes=number_of_cores) as pool:
        pool.map(mp_cal_performance, cal_performance_list)

    df_backtest_result = pd.DataFrame(list(manager_list)).sort_values(by="reference_index")
    df_backtest_result = df_backtest_result.reset_index(drop=True)
    df_backtest_result.to_csv("backtest_result.csv", index=False)


def plot_signal_analysis(py_filename, output_folder, start_date, end_date, para_dict, signal_settings):
    return signals.Signals(
        py_filename,
        output_folder,
        start_date,
        end_date,
        para_dict,
        generate_filepath,
        signal_settings,
    )


def plot(mode, all_para_combination=None, subchart_settings=None, number_of_curves: int = 20, risk_free_rate="geometric_mean"):
    if all_para_combination is None:
        all_para_combination = {}

    if subchart_settings is None:
        subchart_settings = {
            "histogram_period": [1, 3, 5, 10, 20],
            "subchart_1": ["volume", "line"],
        }

    if mode == "equity_curves":
        result_df = pd.read_csv("backtest_result.csv")
        return equity_curves.Plot(all_para_combination, result_df, subchart_settings, number_of_curves)

    if mode == "aggregate":
        return aggregate.Aggregate(risk_free_rate)

    if mode == "signal_analysis":
        return signals.Signals(all_para_combination, generate_filepath, subchart_settings)

    raise ValueError(f"Unknown plot mode: {mode}")


def get_latest_fed_fund_rate() -> float:
    url = "https://fred.stlouisfed.org/series/FEDFUNDS"
    page = requests.get(url, timeout=30)
    page.raise_for_status()

    soup = BeautifulSoup(page.content, "html.parser")
    fed_funds_rate = soup.find("span", class_="series-meta-observation-value").text

    rate = round(float(fed_funds_rate), 2)
    print("Latest Federal Funds Rate:", rate, "%")
    return rate


def get_geometric_mean_of_yearly_rate(start_year: int, end_year: int) -> float:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    data = response.text.split("\n")[:-1]
    data = [row.split(",") for row in data]
    df = pd.DataFrame(data[1:], columns=data[0])
    df.columns = ["date", "risk_free_rate"]
    df["date"] = pd.to_datetime(df["date"])
    df["risk_free_rate"] = pd.to_numeric(df["risk_free_rate"], errors="coerce")
    df = df.dropna(subset=["risk_free_rate"])

    yearly = df.resample("YE", on="date").mean().round(3)
    yearly = yearly[(yearly.index.year >= start_year) & (yearly.index.year <= end_year)]

    fed_fund_rate_geometric_mean = float(np.exp(np.log(yearly["risk_free_rate"]).mean()))
    fed_fund_rate_geometric_mean = round(fed_fund_rate_geometric_mean, 2)

    print(
        f"Federal Funds Rate Geometric mean from {start_year} to {end_year}: "
        f"{fed_fund_rate_geometric_mean} %"
    )
    return fed_fund_rate_geometric_mean


def calculate_mdd(df: pd.DataFrame, col: str):
    roll_max = df[col].cummax()
    daily_drawdown = df[col] / roll_max - 1.0
    max_daily_drawdown = daily_drawdown.cummin()
    return float(max_daily_drawdown.min()), float((df[col] - roll_max).min())


def calculate_win_rate_info(df: pd.DataFrame):
    num_of_trade = (df["action"] == "open").sum()
    num_of_loss = (df["pnl"] < 0).sum()
    num_of_win = num_of_trade - num_of_loss

    if num_of_trade > 0:
        win_rate = round(100 * num_of_win / num_of_trade, 2)
        loss_rate = round(100 * num_of_loss / num_of_trade, 2)
    else:
        win_rate = "--"
        loss_rate = "--"

    return int(num_of_trade), int(num_of_loss), int(num_of_win), win_rate, loss_rate


def calculate_win_rate(df_csv: pd.DataFrame):
    df = df_csv[["date", "realized_pnl", "action"]].copy()
    df = df[df["action"].notnull()].reset_index(drop=True)
    df = df.loc[df["action"] != ""]  # for parquet

    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["pnl"] = df["realized_pnl"].shift(-1)
    df["year"] = pd.DatetimeIndex(df["date"]).year

    year_list = sorted(df["year"].unique().tolist())

    win_rate_dict = {"Overall": calculate_win_rate_info(df)}
    for year in year_list:
        win_rate_dict[year] = calculate_win_rate_info(df.loc[df["year"] == year])

    return win_rate_dict


def calculate_sharpe_ratio(df: pd.DataFrame, col: str, risk_free_rate):
    holding_period_day = (df.loc[df.index[-1], "date"] - df.loc[df.index[0], "date"]).days
    net_profit = df.at[df.index[-1], col] - df.at[df.index[0], col]
    initial_capital = df.loc[df.index[0], col]

    # avoid power error
    if net_profit < 0 and abs(net_profit) > initial_capital:
        net_profit = -initial_capital

    equity_value_pct_series = df[col].pct_change().dropna()

    return_on_capital = net_profit / initial_capital
    annualized_return = (np.sign(1 + return_on_capital) * np.abs(1 + return_on_capital)) ** (
        365 / holding_period_day
    ) - 1
    annualized_std = equity_value_pct_series.std() * math.sqrt(365)

    if annualized_std > 0:
        annualized_sr = (annualized_return - float(risk_free_rate) / 100) / annualized_std
    else:
        annualized_sr = 0

    return (
        net_profit,
        holding_period_day,
        round(100 * return_on_capital, 2),
        round(100 * annualized_return, 2),
        round(100 * annualized_std, 2),
        round(annualized_sr, 2),
    )


def resample_summary_to_daily(para_combination: dict, folder: str = "") -> pd.DataFrame:
    start_date = para_combination["start_date"]
    end_date = para_combination["end_date"]
    start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")

    sec_profile = para_combination["sec_profile"]
    sectype = sec_profile["sectype"]

    code = para_combination["code"]
    lot_size = sec_profile["lot_size_dict"][code]

    intraday = para_combination["intraday"]
    freq = para_combination["freq"]
    file_format = para_combination["file_format"]

    if sectype == "FUT":
        # normalize margin_req key if it appears under a variant name
        if "margin_req" not in sec_profile:
            for key in list(sec_profile.keys()):
                if "margin_req" in key:
                    sec_profile["margin_req"] = sec_profile.pop(key)
                    break
        multiplier = sec_profile["multiplier"]
    elif sectype == "STK":
        multiplier = 1
    else:
        multiplier = 1

    # Read backtest data (may be empty)
    save_path = generate_filepath(para_combination=para_combination, folder=folder)
    if file_format == "parquet":
        df_bt = pl.read_parquet(save_path)
    else:
        df_bt = pl.read_csv(save_path, try_parse_dates=True)

    if len(df_bt) == 0:
        initial_capital = sec_profile["initial_capital"]
    else:
        realized_sum = pl.sum(df_bt.get_column("realized_pnl"))
        if realized_sum is None:
            initial_capital = sec_profile["initial_capital"]
        else:
            initial_capital = df_bt.row(-1, named=True)["equity_value"] - realized_sum

    last_realized_capital = initial_capital

    df_trades = (
        df_bt.lazy()
        .filter(
            (pl.col("action") != "")
            & (pl.col("action").is_not_null())
            & (pl.col("date") >= start_dt)
            & (pl.col("date") <= end_dt)
        )
        .select(["date", "action", "open_price", "close", "commission", "num_of_share", "realized_pnl"])
        .with_columns(pl.col("date").shift(1).alias("open_date"))
        .with_columns(pl.col("open_price").shift(1).alias("open_price"))
        .with_columns(pl.col("num_of_share").shift(1).alias("num_of_share"))
        .filter(pl.col("action") != "open")
        .collect()
    ).to_pandas()

    # Read source data
    data_folder = para_combination["data_folder"]
    data_path = os.path.join(folder, data_folder, f"{code}_{freq}.{file_format}")

    if file_format == "parquet":
        df_src = pl.read_parquet(data_path)
    else:
        df_src = pl.read_csv(data_path, try_parse_dates=True)

    df_daily = (
        df_src.lazy()
        .select(["datetime", "open", "high", "low", "close", "volume"])
        .sort("datetime")
        .groupby_dynamic("datetime", every="1d")
        .agg(
            [
                pl.col("open").first(),
                pl.col("high").max(),
                pl.col("low").min(),
                pl.col("close").last(),
                pl.col("volume").sum(),
            ]
        )
        .filter((pl.col("datetime") >= start_dt) & (pl.col("datetime") <= end_dt))
        .with_columns(pl.col("datetime").alias("date"))
        .with_columns(
            [
                (pl.col("close") * (initial_capital / df_src.head(1)["close"][0])).alias("bah"),
                pl.lit(None).alias("equity_value"),
                pl.lit(None).alias("action"),
                pl.lit(None).alias("realized_pnl"),
                pl.lit(None).alias("unrealized_pnl"),
                pl.lit(None).alias("signal_value"),
                pl.lit(None).alias("commission"),
            ]
        )
        .collect()
    ).to_pandas()

    df_daily.index = pd.to_datetime(df_daily["datetime"], format="%Y-%m-%d")

    # Apply PnL
    apply_pnl.last_realized_capital = last_realized_capital
    apply_pnl.multiplier = multiplier
    apply_pnl.df_daily = df_daily
    df_trades.apply(apply_pnl, axis=1)

    df_daily = apply_pnl.df_daily.reset_index(drop=True)
    df_daily_equity.last_equity_value = initial_capital
    df_daily["equity_value"] = df_daily.apply(df_daily_equity, axis=1).ffill()

    return df_daily


def cal_performance(para_combination: dict) -> dict:
    start_date = para_combination["start_date"]
    end_date = para_combination["end_date"]
    risk_free_rate = para_combination["risk_free_rate"]

    intraday = para_combination["intraday"]
    summary_mode = para_combination["summary_mode"]
    file_format = para_combination["file_format"]

    if intraday or summary_mode:
        df_daily = resample_summary_to_daily(para_combination=para_combination)
    else:
        save_path = generate_filepath(para_combination=para_combination)
        if file_format == "parquet":
            df_backtest = pd.read_parquet(save_path)
        else:
            df_backtest = pd.read_csv(save_path, index_col=0)

        df_backtest["date"] = pd.to_datetime(df_backtest["date"], format="%Y-%m-%d")
        df_backtest = df_backtest.loc[(df_backtest["date"] >= start_date) & (df_backtest["date"] <= end_date)]
        df_daily = df_backtest.reset_index(drop=True)

    df = df_daily

    equity_value_column = df["equity_value"].to_numpy()
    no_trade = (equity_value_column[0] == equity_value_column).all()

    result_dict: dict = {}

    start_year = datetime.datetime.strptime(start_date, "%Y-%m-%d").year
    end_year = datetime.datetime.strptime(end_date, "%Y-%m-%d").year
    year_list = list(range(start_year, end_year + 1))
    for y in year_list:
        result_dict[str(y)] = []

    if no_trade:
        result_dict.update(
            {
                "holding_period_day": 0,
                "total_commission": 0,
                "net_profit": 0,
                "return_on_capital": 0,
                "annualized_return": 0,
                "annualized_std": 0,
                "annualized_sr": 0,
                "mdd_dollar": 0,
                "mdd_pct": 0,
                "num_of_trade": 0,
                "win_rate": 0,
                "loss_rate": 0,
                "net_profit_to_mdd": np.inf,
                "cov_count": 0,
                "cov_return": 0,
            }
        )
        for year in year_list:
            result_dict[str(year)] = 0
            result_dict[f"{year}_win_rate"] = "--"
            result_dict[f"{year}_return"] = 0
    else:
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
        df["year"] = pd.DatetimeIndex(df["date"]).year

        net_profit, holding_period_day, return_on_capital, annualized_return, annualized_std, annualized_sr = (
            calculate_sharpe_ratio(df, "equity_value", risk_free_rate)
        )
        mdd_pct, mdd_dollar = calculate_mdd(df, "equity_value")
        mdd_pct *= -100
        mdd_dollar *= -1

        save_path = generate_filepath(para_combination=para_combination)
        if file_format == "parquet":
            df_backtest = pd.read_parquet(save_path)
        else:
            df_backtest = pd.read_csv(save_path, index_col=0)

        win_rate_dict = calculate_win_rate(df_backtest)
        num_of_trade, num_of_loss, num_of_win, win_rate, loss_rate = win_rate_dict["Overall"]

        total_commission = df["commission"].sum()

        result_dict.update(
            {
                "holding_period_day": holding_period_day,
                "total_commission": total_commission,
                "net_profit": net_profit,
                "return_on_capital": return_on_capital,
                "annualized_return": annualized_return,
                "annualized_std": annualized_std,
                "annualized_sr": annualized_sr,
                "mdd_dollar": mdd_dollar,
                "mdd_pct": mdd_pct,
                "num_of_trade": num_of_trade,
                "win_rate": win_rate,
                "loss_rate": loss_rate,
                "net_profit_to_mdd": np.inf if mdd_dollar == 0 else net_profit / mdd_dollar,
            }
        )

        # trades and win rate by year
        year_count = []
        for year in year_list:
            try:
                result_dict[str(year)] = win_rate_dict[year][0]
                result_dict[f"{year}_win_rate"] = win_rate_dict[year][3]
            except Exception:
                result_dict[str(year)] = 0
                result_dict[f"{year}_win_rate"] = "--"
            year_count.append(result_dict[str(year)])

        std_dev = np.std(year_count)
        mean = np.mean(year_count)
        if std_dev != 0 and not np.isnan(std_dev) and mean != 0 and not np.isnan(mean):
            result_dict["cov_count"] = round(std_dev / mean, 3)
        else:
            result_dict["cov_count"] = 0

        # performance by year
        first_equity_value = 0
        last_equity_value = 0
        year_return_list = []

        for year in year_list:
            if not df.loc[df["year"] == year].empty:
                if first_equity_value == 0:
                    first_equity_value = df.loc[df["year"] == year].iloc[0].equity_value

                last_equity_value = df.loc[df["year"] == year].iloc[-1].equity_value
                yearly_return = (last_equity_value - first_equity_value) / first_equity_value

                if np.isnan(yearly_return):
                    result_dict[f"{year}_return"] = 0
                    year_return_list.append(0)
                else:
                    pct = int(yearly_return * 100)
                    result_dict[f"{year}_return"] = pct
                    year_return_list.append(pct)
            else:
                result_dict[f"{year}_return"] = "-----"
                year_return_list.append(0)

            first_equity_value = last_equity_value

        return_year_std = np.std(year_return_list)
        return_year_mean = np.mean(year_return_list)
        result_dict["cov_return"] = 0 if return_year_mean == 0 else round(return_year_std / return_year_mean, 3)

    result_dict["risk_free_rate"] = risk_free_rate

    # Buy-and-hold performance
    bah_net_return, _, bah_return, bah_annualized_return, bah_annualized_std, bah_annualized_sr = calculate_sharpe_ratio(
        df, "close", risk_free_rate
    )
    initial_capital = df.loc[df.index[0], "equity_value"]
    df["bah_equity_curve"] = df["close"] * initial_capital // df.loc[df.index[0], "close"]

    bah_mdd_pct, bah_mdd_dollar = calculate_mdd(df, "bah_equity_curve")
    bah_mdd_pct *= -100
    bah_mdd_dollar *= -1

    result_dict.update(
        {
            "bah_return": bah_return,
            "bah_annualized_return": bah_annualized_return,
            "bah_annualized_std": bah_annualized_std,
            "bah_annualized_sr": bah_annualized_sr,
            "bah_mdd_dollar": bah_mdd_dollar,
            "bah_mdd_pct": bah_mdd_pct,
            "return_to_bah": result_dict.get("return_on_capital", 0) - bah_return,
        }
    )

    return result_dict
