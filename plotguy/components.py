import ast
import datetime
import math
import os

import numpy as np
import pandas as pd

import dash_bootstrap_components as dbc
from dash import dcc, html

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import dash_dangerously_set_inner_html  # for all Plot Class
import plotguy


class Components:
    chart_bg = "#1f2c56"

    # Static UI bits (overridden in __init__ where needed)
    sort_method_dropdown = html.Div()

    filter_dropdown = html.Div(
        id="filter_dropdown",
        children=dbc.Select(
            id="filter_name",
            placeholder="Select Filter",
            options=[
                {"label": "Exclude Stock", "value": "exclude"},
                {"label": "Return on Capital >", "value": "return_on_capital>"},
                {"label": "Return on Capital <", "value": "return_on_capital<"},
                {"label": "Annualized Return >", "value": "annualized_return>"},
                {"label": "Annualized Return <", "value": "annualized_return<"},
                {"label": "Return-BaH % Diff. >", "value": "return_to_bah>"},
                {"label": "Return-BaH % Diff. <", "value": "return_to_bah<"},
                {"label": "Sharpe Ratio >", "value": "annualized_sr>"},
                {"label": "Sharpe Ratio <", "value": "annualized_sr<"},
                {"label": "MDD Percentage >", "value": "mdd_pct>"},
                {"label": "MDD Percentage <", "value": "mdd_pct<"},
                {"label": "Trade Count >", "value": "num_of_trade>"},
                {"label": "Trade Count <", "value": "num_of_trade<"},
                {"label": "COV (Count) >", "value": "cov_count>"},
                {"label": "COV (Count) <", "value": "cov_count<"},
                {"label": "COV (Return) >", "value": "cov_return>"},
                {"label": "COV (Return) <", "value": "cov_return<"},
                {"label": "Win Rate >", "value": "win_rate>"},
                {"label": "Win Rate <", "value": "win_rate<"},
            ],
            style={"border-radius": "5px", "font-size": "12px"},
        ),
        style={"padding-left": "15px", "width": "175px"},
    )

    filter_dropdown_disabled = html.Div(
        id="filter_dropdown",
        children=dbc.Select(
            id="filter_name",
            disabled=True,
            placeholder="Select Filter",
            style={"border-radius": "5px", "font-size": "12px", "backgroundColor": "Gray"},
        ),
        style={"padding-left": "15px", "width": "180px"},
    )

    filter_input = dbc.Input(
        id="filter_input",
        value=None,
        size="md",
        type="text",
        style={
            "width": "50px",
            "margin-right": "5px",
            "border-radius": "3px",
            "padding": "6px 5px",
            "font-size": "12px",
        },
    )

    filter_input_disabled = dbc.Input(
        id="filter_input",
        value=None,
        size="md",
        disabled=True,
        style={
            "width": "50px",
            "margin-right": "5px",
            "border-radius": "3px",
            "padding": "6px 5px",
            "font-size": "12px",
            "backgroundColor": "Gray",
        },
    )

    add_button_style = {
        "margin-left": "50px",
        "width": "150px",
        "backgroundColor": "blue",
        "border-radius": "5px",
        "text-align": "center",
        "cursor": "pointer",
        "font-size": "13px",
        "height": "30px",
    }

    add_button_style_disabled = {
        "margin-left": "50px",
        "width": "150px",
        "color": "Silver",
        "backgroundColor": "Gray",
        "border-radius": "5px",
        "text-align": "center",
        "font-size": "13px",
        "height": "30px",
    }

    filter_options = {
        "num_of_trade": "Trade Count",
        "return_on_capital": "Return on Capital",
        "annualized_return": "Annualized Return",
        "annualized_sr": "Sharpe Ratio",
        "mdd_pct": "MDD Percentage",
        "cov_count": "COV (Count)",
        "cov_return": "COV (Return)",
        "win_rate": "Win Rate",
        "return_to_bah": "Return/BnH % Diff.",
        "exclude": "Exclude",
    }

    def __init__(self, number_of_curves: int):
        self.sort_method_dropdown = html.Div(
            dbc.Select(
                id="sort_method",
                placeholder="Select Sorting Method",
                value="Top Net Profit",
                options=[
                    {"label": f"Top {number_of_curves} Net Profit", "value": "Top Net Profit"},
                    {
                        "label": f"Top {number_of_curves} Return-BaH % Difference",
                        "value": "Top Return to BaH Ratio",
                    },
                    {"label": f"Top {number_of_curves} Sharpe Ratio", "value": "Top Sharpe Ratio"},
                ],
                style={"border-radius": "5px", "font-size": "12px"},
            ),
            style={"padding-left": "15px", "width": "235px"},
        )

    # -------------------------
    # Chart helpers
    # -------------------------
    def empty_line_chart(self):
        fig_line = px.line()
        fig_line.update_layout(
            plot_bgcolor=self.chart_bg,
            paper_bgcolor=self.chart_bg,
            margin=dict(l=85, r=60, t=30, b=40),
            height=500,
            font={"color": self.chart_bg},
        )
        fig_line.update_xaxes(showline=True, zeroline=False, linecolor="#979A9A", gridcolor=self.chart_bg)
        fig_line.update_yaxes(showline=True, zeroline=False, linecolor="#979A9A", gridcolor=self.chart_bg)
        return fig_line

    # -------------------------
    # UI builders
    # -------------------------
    def update_checkbox_div(self, para_dict, result_df: pd.DataFrame):
        """Build checklist UI for each parameter in para_dict, based on unique values in result_df."""
        checkbox_values = {}
        for key in para_dict:
            unique_values = list(dict.fromkeys(result_df[key].tolist()))
            try:
                unique_values.sort()
            except Exception:
                pass
            checkbox_values[key] = unique_values

        checkbox_div = []
        code_list = []

        for i, para_name in enumerate(para_dict):
            options = checkbox_values.get(para_name, [])

            if para_name == "code":
                formatted = []
                for option in options:
                    if isinstance(option, int):
                        formatted.append(str(option).zfill(5))
                    else:
                        formatted.append(option)
                options = formatted
                code_list = options

            if options == [False, True]:
                checklist = dcc.Checklist(
                    [str(v) for v in options],
                    [str(v) for v in options],
                    inline=True,
                    id={"type": "para-checklist", "index": i},
                    labelStyle={},
                    inputStyle={"margin-left": "10px", "margin-right": "3px"},
                )
            else:
                base_checklist = dcc.Checklist(
                    options,
                    options,
                    inline=True,
                    id={"type": "para-checklist", "index": i},
                    labelStyle={"font-size": "12px"},
                    inputStyle={
                        "margin-left": "10px",
                        "margin-right": "3px",
                        "background-color": "red",
                        "vertical-align": "middle",
                        "position": "relative",
                        "bottom": ".15em",
                    },
                )

                if para_name == "code":
                    checklist = dbc.Row(
                        [
                            dcc.Checklist(
                                id="all-or-none",
                                options=[{"label": "Select All", "value": "All"}],
                                labelStyle={"font-size": "12px"},
                                inputStyle={
                                    "margin-left": "10px",
                                    "margin-right": "3px",
                                    "background-color": "red",
                                    "vertical-align": "middle",
                                    "position": "relative",
                                    "bottom": ".15em",
                                },
                            ),
                            html.Div(style={"height": "5px"}),
                            base_checklist,
                        ]
                    )
                else:
                    checklist = base_checklist

            row = html.Div(
                dbc.Row(
                    [
                        html.Div(para_name),
                        html.Div(style={"height": "5px"}),
                        html.Div(checklist),
                        html.Div(style={"height": "5px"}),
                    ]
                ),
                style={"padding": "0px 20px", "font-size": "12px"},
            )
            checkbox_div.append(row)

        return checkbox_div, code_list

    def update_stat_div(self, period, pct_mean, rise_mean, fall_mean):
        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(width=2),
                        dbc.Col(f"{period[0]} Days", width=2),
                        dbc.Col(f"{period[1]} Days", width=2),
                        dbc.Col(f"{period[2]} Days", width=2),
                        dbc.Col(f"{period[3]} Days", width=2),
                        dbc.Col(f"{period[4]} Days", width=2),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col("pct_change", width=2),
                        dbc.Col(f"{pct_mean[0]}%", width=2),
                        dbc.Col(f"{pct_mean[1]}%", width=2),
                        dbc.Col(f"{pct_mean[2]}%", width=2),
                        dbc.Col(f"{pct_mean[3]}%", width=2),
                        dbc.Col(f"{pct_mean[4]}%", width=2),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col("max_rise", width=2),
                        dbc.Col(f"{rise_mean[0]}%", width=2),
                        dbc.Col(f"{rise_mean[1]}%", width=2),
                        dbc.Col(f"{rise_mean[2]}%", width=2),
                        dbc.Col(f"{rise_mean[3]}%", width=2),
                        dbc.Col(f"{rise_mean[4]}%", width=2),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col("max_fall", width=2),
                        dbc.Col(f"{fall_mean[0]}%", width=2),
                        dbc.Col(f"{fall_mean[1]}%", width=2),
                        dbc.Col(f"{fall_mean[2]}%", width=2),
                        dbc.Col(f"{fall_mean[3]}%", width=2),
                        dbc.Col(f"{fall_mean[4]}%", width=2),
                    ]
                ),
            ]
        )

    def generate_radioitems(self, para_dict):
        radioitems_div = []
        for i, key in enumerate(para_dict):
            options = para_dict[key]

            radioitems_div.append(html.Div(key, style={"color": "Yellow", "font-size": "14px"}))
            radioitems_div.append(
                dcc.RadioItems(
                    id={"type": "para_radioitems", "index": i},
                    options=[{"label": k, "value": k} for k in options],
                    labelStyle={"font-size": "13px"},
                    inputStyle={"margin-left": "10px", "margin-right": "5px"},
                )
            )
            radioitems_div.append(html.Div(html.Img(), style={"height": "10px"}))
        return radioitems_div

    def selection_title(self, para_dict, values):
        title = []
        for i, key in enumerate(para_dict):
            title.append(
                dbc.Row(
                    [
                        dbc.Col(key, width=6),
                        dbc.Col(f"{values[i]}", style={"text-align": "center"}, width=6),
                    ]
                )
            )
        return title

    # -------------------------
    # Performance matrix
    # -------------------------
    def update_performance_matrix(self, start_date, end_date, df, para_dict, risk_free_rate, ref_code):
        per_col1 = []
        per_col2 = []
        per_col_1_1 = []
        per_col_1_2 = []
        per_col_2_1 = [html.Div(html.Img())]
        per_col_2_2 = [html.Div("Selected", style={"text-align": "center", "font-weight": "bold", "color": "yellow"})]
        per_col_2_3 = [html.Div("BaH", style={"text-align": "center", "font-weight": "bold", "color": "yellow"})]

        keys = [
            "num_of_trade",
            "net_profit",
            "net_profit_to_mdd",
            "return_to_bah",
            "win_rate",
            "cov_count",
            "cov_return",
            "total_commission",
            "return_on_capital",
            "annualized_return",
            "annualized_std",
            "annualized_sr",
            "mdd_dollar",
            "mdd_pct",
            "bah_return",
            "bah_annualized_return",
            "bah_annualized_std",
            "bah_annualized_sr",
            "bah_mdd_dollar",
            "bah_mdd_pct",
        ]

        # Best-effort formatting (df behaves like a dict-like row)
        try:
            df["net_profit"] = "{:,}".format(int(round(df["net_profit"], 0)))
            df["net_profit_to_mdd"] = "inf" if df["net_profit_to_mdd"] == np.inf else round(df["net_profit_to_mdd"], 2)
            df["total_commission"] = "{:,}".format(int(round(df["total_commission"], 0)))

            df["mdd_dollar"] = "{:,}".format(int(round(df["mdd_dollar"], 0)))
            df["mdd_pct"] = "{:.0%}".format(df["mdd_pct"] / 100)
            df["return_on_capital"] = "{:.0%}".format(df["return_on_capital"] / 100)
            df["annualized_return"] = "{:.0%}".format(df["annualized_return"] / 100)
            df["annualized_std"] = "{:.0%}".format(df["annualized_std"] / 100)

            df["cov_count"] = round(df["cov_count"], 2)
            df["cov_return"] = round(df["cov_return"], 2)

            try:
                df["win_rate"] = "{:.0%}".format(float(df["win_rate"]) / 100)
            except Exception:
                df["win_rate"] = "--"

            df["return_to_bah"] = "{:.0%}".format(df["return_to_bah"] / 100)

            df["bah_mdd_dollar"] = "{:,}".format(int(round(df["bah_mdd_dollar"], 0)))
            df["bah_mdd_pct"] = "{:.0%}".format(df["bah_mdd_pct"] / 100)
            df["bah_return"] = "{:.0%}".format(df["bah_return"] / 100)
            df["bah_annualized_return"] = "{:.0%}".format(df["bah_annualized_return"] / 100)
            df["bah_annualized_std"] = "{:.0%}".format(df["bah_annualized_std"] / 100)
        except Exception:
            pass

        per_col_1_1.extend(
            [
                html.Div("Number of Trade"),
                html.Div("Net Profit"),
                html.Div("Net Profit/MDD"),
                html.Div("Return-BaH % Diff."),
                html.Div("Win Rate"),
                html.Div("COV (Count)"),
                html.Div("COV (Return)"),
                html.Div("Total Commission"),
                html.Div("Risk Free Rate"),
            ]
        )

        per_col_2_1.extend(
            [
                html.Div("Return on Capital"),
                html.Div("Ann. Return"),
                html.Div("Ann. Std"),
                html.Div("Ann. Sharpe Ratio"),
                html.Div("MDD Dollar"),
                html.Div("MDD Percentage"),
            ]
        )

        for i in range(8):
            per_col_1_2.append(html.Div(df[keys[i]], style={"text-align": "center"}))

        try:
            per_col_1_2.append(html.Div("{:.2%}".format(risk_free_rate / 100), style={"text-align": "center"}))
        except Exception:
            per_col_1_2.append(html.Div("-----", style={"text-align": "center"}))

        for i in range(8, 14):
            per_col_2_2.append(html.Div(str(df[keys[i]]), style={"text-align": "center"}))
        for i in range(14, 20):
            per_col_2_3.append(html.Div(str(df[keys[i]]), style={"text-align": "center"}))

        per_col1.append(
            dbc.Row(
                [
                    dbc.Col(html.Div(per_col_1_1), width=6),
                    dbc.Col(per_col_1_2, style={"padding": "0"}, width=6),
                ]
            )
        )
        per_col2.append(
            dbc.Row(
                [
                    dbc.Col(html.Div(per_col_2_1), width=6),
                    dbc.Col(per_col_2_2, style={"padding": "0"}, width=3),
                    dbc.Col(per_col_2_3, style={"padding": "0"}, width=3),
                ]
            )
        )

        start_year = datetime.datetime.strptime(start_date, "%Y-%m-%d").year
        end_year = datetime.datetime.strptime(end_date, "%Y-%m-%d").year
        year_list = list(range(start_year, end_year + 1))

        year_col1 = [
            dbc.Row(
                [
                    dbc.Col(style={"margin-left": "15px"}, width=2),
                    dbc.Col("Count", style={"text-align": "center", "font-weight": "bold", "color": "yellow"}, width=3),
                    dbc.Col(
                        "WinRate", style={"text-align": "center", "font-weight": "bold", "color": "yellow"}, width=3
                    ),
                    dbc.Col(
                        "Return", style={"text-align": "center", "font-weight": "bold", "color": "yellow"}, width=3
                    ),
                ],
                style={"font-size": "11px"},
            )
        ]

        for y in year_list:
            try:
                win_rate = "{:.0%}".format(int(float(df[f"{y}_win_rate"])) / 100)
            except Exception:
                win_rate = "-----"

            try:
                year_return = "{:.0%}".format(int(float(df[f"{y}_return"])) / 100)
            except Exception:
                year_return = "-----"

            year_col1.append(
                dbc.Row(
                    [
                        dbc.Col(y, style={"margin-left": "15px"}, width=2),
                        dbc.Col(df[str(y)], style={"text-align": "center"}, width=3),
                        dbc.Col(win_rate, style={"text-align": "center"}, width=3),
                        dbc.Col(year_return, style={"text-align": "center"}, width=3),
                    ]
                )
            )

        # Selected Equity Curve title block
        title_rows = []
        for key in para_dict:
            title_rows.append(dbc.Row([dbc.Col(key, width=6), dbc.Col(f"{df[key]}", style={"text-align": "center"}, width=6)]))

        matrix_div = html.Div(
            [
                html.Div(style={"height": "8px"}),
                html.Div(
                    [
                        html.Div("Selected Curve", style={"color": "Cyan", "display": "inline", "font-size": "15px"}),
                        html.Div(
                            "",
                            id="save_status",
                            style={
                                "color": "Cyan",
                                "display": "inline",
                                "padding-left": "5px",
                                "font-size": "15px",
                            },
                        ),
                    ]
                ),
                dbc.Row(html.Div(title_rows), style={"font-size": "11px", "padding-left": "1px"}),
                html.Div(style={"height": "15px"}),
                html.Div("Performance", style={"color": "Cyan", "font-size": "15px"}),
                html.Div(children=per_col1, style={"font-size": "11px", "padding-left": "1px"}),
                html.Div(style={"height": "15px"}),
                html.Div("Comparison", style={"color": "Cyan", "font-size": "15px"}),
                html.Div(
                    children=per_col2,
                    style={"font-size": "11px", "padding-right": "10px", "padding-left": "1px"},
                ),
                html.Div(style={"height": "15px"}),
                html.Div("Performance by Year", style={"color": "Cyan", "font-size": "15px"}),
                html.Div(style={"height": "5px"}),
                html.Div(children=year_col1, style={"font-size": "11px"}),
                html.Div(style={"height": "10px"}),
                html.Div(f"Ref: {ref_code}", style={"font-size": "11px"}),
                html.Div(style={"height": "20px"}),
                html.Div(
                    [
                        html.Div(style={"height": "5px"}),
                        html.Div("", id="save_string"),
                        html.Div(style={"height": "5px"}),
                    ],
                    id="save_button",
                ),
            ],
            style={"padding": "0px"},
        )

        return matrix_div

    def update_filter_div(self, filter_list):
        filter_button = []
        for i, element in enumerate(filter_list):
            key, op, val = element[0], element[1], element[2]

            filter_full = [
                html.Div(self.filter_options[key], style={"margin-right": "15px", "display": "inline"}),
                html.Div(op, style={"margin-right": "15px", "display": "inline"}),
                html.Div(val, style={"margin-right": "15px", "display": "inline"}),
            ]

            filter_button.append(
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(filter_full, style={"font-size": "12px", "padding": "0px", "margin": "0px"}),
                            width=10,
                        ),
                        dbc.Col(
                            html.Div(
                                children=html.Div("✗", style={"padding": "0px", "margin": "0px"}),
                                id=f"button_{i}",
                                n_clicks=i,
                                style={
                                    "font-size": "12px",
                                    "backgroundColor": "rgba(0, 0, 0, 0)",
                                    "border": "0px black solid",
                                    "padding": "0px",
                                    "padding-bottom": "10px",
                                    "margin": "0px",
                                    "width": "5px",
                                    "cursor": "pointer",
                                },
                            ),
                            width=2,
                        ),
                    ]
                )
            )

        for i in range(len(filter_list), 10):
            filter_button.append(html.Div(id=f"button_{i}", n_clicks=i))

        return filter_button

    # -------------------------
    # Sorting + colors
    # -------------------------
    def sort_method_df(self, sort_method, result_df: pd.DataFrame, number_of_curves: int):
        if sort_method == "Top Net Profit":
            df_sorted = result_df.sort_values(by="net_profit", ascending=False).head(number_of_curves).copy()
        elif sort_method == "Top Sharpe Ratio":
            df_sorted = result_df.sort_values(by="annualized_sr", ascending=False).head(number_of_curves).copy()
        elif sort_method == "Top Return to BaH Ratio":
            df_sorted = result_df.sort_values(by="return_to_bah", ascending=False).head(number_of_curves).copy()
        else:
            df_sorted = result_df.copy()

        df_sorted = df_sorted.reset_index(drop=True)

        line_colour = []
        for c in range(len(df_sorted)):
            profile = c % 6
            degree = (c // 6) / math.ceil(len(df_sorted) / 6)
            line_colour.append(self.assign_colour(profile, degree))
        df_sorted["line_colour"] = line_colour

        return df_sorted

    def assign_colour(self, profile: int, degree: float) -> str:
        if profile == 0:
            rgb = (0, int(252 - 252 * degree), 252)
        elif profile == 1:
            rgb = (int(252 - 252 * degree), 252, 0)
        elif profile == 2:
            rgb = (252, 0, int(252 - 252 * degree))
        elif profile == 3:
            rgb = (0, 252, int(252 * degree))
        elif profile == 4:
            rgb = (252, int(252 * degree), 0)
        else:
            rgb = (int(252 * degree), 0, 252)
        return "rgb" + str(rgb)

    # -------------------------
    # Chart 3 (candlestick)
    # -------------------------
    def generate_chart_3(self, para_combination, chart3_start, chart3_end, freq, file_format):
        data_folder = para_combination["data_folder"]
        code = para_combination["code"]
        data_path = os.path.join(data_folder, f"{code}_{freq}.{file_format}")

        if file_format == "parquet":
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path, index_col=0)

        df = df.reset_index(drop=False)
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
        df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M:%S")
        df = df.loc[(df["date"] >= chart3_start) & (df["date"] <= chart3_end)]

        dt_all = pd.date_range(start=df["datetime"].iloc[0], end=df["datetime"].iloc[-1], freq=freq)
        dt_obs = [d.strftime("%Y-%m-%d %H:%M:%S") for d in df["datetime"]]
        dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d %H:%M:%S").tolist() if d not in dt_obs]

        save_path = plotguy.generate_filepath(para_combination=para_combination)
        if file_format == "parquet":
            df_signal_list = pd.read_parquet(save_path)
        else:
            df_signal_list = pd.read_csv(save_path, index_col=0)

        df_signal_list["date"] = pd.to_datetime(df_signal_list["date"], format="%Y-%m-%d")
        try:
            df_signal_list["datetime"] = pd.to_datetime(df_signal_list["datetime"], format="%Y-%m-%d %H:%M:%S")
        except Exception:
            df_signal_list = df_signal_list.reset_index()
            df_signal_list["datetime"] = pd.to_datetime(df_signal_list["datetime"], format="%Y-%m-%d %H:%M:%S")

        df_signal_list = df_signal_list.loc[
            (df_signal_list["date"] >= chart3_start) & (df_signal_list["date"] <= chart3_end)
        ]

        df_open = df_signal_list.loc[df_signal_list["action"] == "open"]
        df_profit = df_signal_list.loc[df_signal_list["action"] == "profit_target"]
        df_stop_loss = df_signal_list.loc[df_signal_list["action"] == "stop_loss"]
        df_close = df_signal_list.loc[df_signal_list["action"] == "close_logic"]

        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df["datetime"],
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                    increasing_line_color="white",
                )
            ]
        )

        fig.update_layout(title={"text": "Candlestick Chart", "font": {"size": 12}})
        fig.update_xaxes(rangebreaks=[dict(dvalue=15 * 60 * 1000, values=dt_breaks)])

        marker_base = dict(color="rgba(0, 0, 0, 0)", size=18, line=dict(width=2.5))
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=df_open["datetime"],
                y=df_open["close"],
                marker={**marker_base, "line": dict(color="yellow", width=2.5)},
                name="Open",
            )
        )
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=df_profit["datetime"],
                y=df_profit["close"],
                marker={**marker_base, "line": dict(color="Cyan", width=2.5)},
                name="Profit Target",
            )
        )
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=df_stop_loss["datetime"],
                y=df_stop_loss["close"],
                marker={**marker_base, "line": dict(color="Cyan", width=2.5)},
                name="Stop Loss",
            )
        )
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=df_close["datetime"],
                y=df_close["close"],
                marker={**marker_base, "line": dict(color="Cyan", width=2.5)},
                name="Close Logic",
            )
        )

        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.update_layout(
            plot_bgcolor=self.chart_bg,
            paper_bgcolor=self.chart_bg,
            height=500,
            margin=dict(l=85, r=25, t=60, b=0),
            showlegend=False,
            font={"color": "white", "size": 10.5},
            yaxis={"title": "Equity"},
            xaxis={"title": ""},
        )
        fig.update_xaxes(showline=True, zeroline=False, linecolor="#979A9A", gridcolor=self.chart_bg)
        fig.update_yaxes(showline=True, zeroline=False, linecolor="#979A9A", gridcolor=self.chart_bg)
        return fig

    # -------------------------
    # Chart 1 (multi equity curves)
    # -------------------------
    def generate_chart_1(self, graph_df: pd.DataFrame, all_para_combination, intraday: bool, summary_mode: bool):
        py_filename = all_para_combination[0]["py_filename"]
        para_dict = all_para_combination[0]["para_dict"]
        para_key_list = list(para_dict)
        df_all = pd.DataFrame(all_para_combination)

        fig_line = px.line()
        fig_line.update_layout(title={"text": py_filename})
        fig_line.update_xaxes(showline=True, zeroline=False, linecolor="white", gridcolor="rgba(0, 0, 0, 0)")
        fig_line.update_yaxes(showline=True, zeroline=False, linecolor="white", gridcolor="rgba(0, 0, 0, 0)")
        fig_line.update_layout(
            plot_bgcolor=self.chart_bg,
            paper_bgcolor=self.chart_bg,
            height=500,
            margin=dict(l=85, r=25, t=60, b=0),
            showlegend=False,
            font={"color": "white", "size": 10.5},
            yaxis={"title": "Equity"},
            xaxis={"title": ""},
        )

        for i in graph_df.index:
            hovertemplate = "%{x}<br>"

            for key in para_key_list:
                hovertemplate += f"{key} : {graph_df.iloc[i][key]}<br>"
            hovertemplate += "<br>"

            hovertemplate += f"Trade Count : {graph_df.iloc[i]['num_of_trade']}<br>"
            hovertemplate += "Net Profit : " + "{:,}".format(int(round(graph_df.iloc[i]["net_profit"], 0))) + "<br>"
            hovertemplate += "Net Profit to MDD: " + str(round(graph_df.iloc[i]["net_profit_to_mdd"], 2)) + "<br>"
            hovertemplate += (
                "Return-BaH % Diff. : " + "{:.0%}".format(float(graph_df.iloc[i]["return_to_bah"] / 100)) + "<br>"
            )

            try:
                hovertemplate += "Win Rate : " + "{:.0%}".format(float(graph_df.iloc[i]["win_rate"]) / 100) + "<br>"
            except Exception:
                hovertemplate += "Win Rate : --<br>"

            hovertemplate += "COV (Count) : " + str(round(graph_df.iloc[i]["cov_count"], 2)) + "<br>"
            hovertemplate += "COV (Return) : " + str(round(graph_df.iloc[i]["cov_return"], 2)) + "<br>"
            hovertemplate += (
                "Total Commission : " + "{:,}".format(int(round(graph_df.iloc[i]["total_commission"], 2))) + "<br>"
            )

            hovertemplate += "<br>"
            hovertemplate += "Return on Capital : " + "{:.0%}".format(graph_df.iloc[i]["return_on_capital"] / 100) + "<br>"
            hovertemplate += "Annual Return : " + "{:.0%}".format(graph_df.iloc[i]["annualized_return"] / 100) + "<br>"
            hovertemplate += "Sharpe Ratio : " + str(graph_df.iloc[i]["annualized_sr"]) + "<br>"
            hovertemplate += "MDD Percentage : " + "{:.0%}".format(graph_df.iloc[i]["mdd_pct"] / 100) + "<br>"
            hovertemplate += "<br>"

            reference_index = graph_df.iloc[i].reference_index
            para_combination = df_all.loc[df_all["reference_index"] == reference_index].to_dict("records")[0]
            line_colour = graph_df.loc[i].line_colour
            file_format = para_combination["file_format"]

            if intraday or summary_mode:
                df_daily = plotguy.resample_summary_to_daily(para_combination=para_combination)
            else:
                save_path = plotguy.generate_filepath(para_combination=para_combination)
                if file_format == "parquet":
                    df_backtest = pd.read_parquet(save_path)
                else:
                    df_backtest = pd.read_csv(save_path, index_col=0)
                df_daily = df_backtest.copy()

            start_date = para_combination["start_date"]
            end_date = para_combination["end_date"]
            df_daily["date"] = pd.to_datetime(df_daily["date"], format="%Y-%m-%d")
            df_daily = df_daily.loc[(df_daily["date"] >= start_date) & (df_daily["date"] <= end_date)].reset_index(drop=True)

            hovertemplate += "Equity : %{y:,.0f}"

            fig_line.add_trace(
                go.Scatter(
                    mode="lines",
                    hovertemplate=hovertemplate,
                    x=df_daily["date"],
                    y=df_daily["equity_value"],
                    line=dict(color=line_colour, width=1.5),
                    name="",
                )
            )

        return fig_line

    # -------------------------
    # Chart 2 (equity vs BnH + signals)
    # -------------------------
    def prepare_df_chart(self, df: pd.DataFrame):
        initial_value = df.iloc[0].equity_value
        df_chart = df.copy()
        df_chart["date"] = pd.to_datetime(df_chart["date"], format="%Y-%m-%d")
        df_chart["bah"] = df["close"] * (initial_value / df_chart.iloc[0].close)

        open_, stop_loss, close_logic, profit_target = [], [], [], []

        for i, action in enumerate(list(df["action"])):
            if action == "open":
                open_.append(df_chart.iloc[i].bah)
                stop_loss.append(None)
                close_logic.append(None)
                profit_target.append(None)
            elif action == "stop_loss":
                open_.append(None)
                stop_loss.append(df_chart.iloc[i].bah)
                close_logic.append(None)
                profit_target.append(None)
            elif action == "close_logic":
                open_.append(None)
                stop_loss.append(None)
                close_logic.append(df_chart.iloc[i].bah)
                profit_target.append(None)
            elif action == "profit_target":
                open_.append(None)
                stop_loss.append(None)
                close_logic.append(None)
                profit_target.append(df_chart.iloc[i].bah)
            else:
                open_.append(None)
                stop_loss.append(None)
                close_logic.append(None)
                profit_target.append(None)

        df_chart["open"] = open_
        df_chart["stop_loss"] = stop_loss
        df_chart["close_logic"] = close_logic
        df_chart["profit_target"] = profit_target
        return df_chart

    def generate_chart_2_summary(self, para_combination, line_colour):
        df_daily = plotguy.resample_summary_to_daily(para_combination=para_combination)

        title = ""
        para_dict = para_combination["para_dict"]
        for key in para_dict:
            title += f"{key}:{para_combination[key]} "

        start_date = para_combination["start_date"]
        end_date = para_combination["end_date"]
        df_daily["date"] = pd.to_datetime(df_daily["date"], format="%Y-%m-%d")
        df_daily = df_daily.loc[(df_daily["date"] >= start_date) & (df_daily["date"] <= end_date)].reset_index(drop=True)

        df_chart = df_daily.copy()
        row_height = [0.85, 0.15]
        fig_line = make_subplots(rows=2, cols=1, row_heights=row_height, shared_xaxes=True)

        fig_line.update_layout(title={"text": title, "font": {"size": 12}})
        fig_line.add_trace(
            go.Scatter(
                mode="lines",
                hoverinfo="skip",
                x=df_chart["date"],
                y=df_chart["equity_value"],
                line=dict(color=line_colour, width=1),
                name="Strategy Equity",
            )
        )
        fig_line.add_trace(
            go.Scatter(
                mode="lines",
                hoverinfo="skip",
                x=df_chart["date"],
                y=df_chart["bah"],
                line=dict(color="Grey", width=1),
                name="BnH Equity",
            )
        )

        fig_line.add_trace(
            go.Scatter(
                mode="markers",
                hovertemplate="Date : %{x}",
                x=df_chart["date"],
                y=df_chart["signal_value"],
                marker=dict(color="rgba(0, 0, 0, 0)", size=9, line=dict(color="yellow", width=2.5)),
                name="Open",
            )
        )

        fig_line.update_layout(
            plot_bgcolor=self.chart_bg,
            paper_bgcolor=self.chart_bg,
            height=500,
            margin=dict(l=85, r=25, t=60, b=0),
            font={"color": "white", "size": 10.5},
            yaxis={"title": "Equity"},
            xaxis={"title": ""},
        )
        fig_line.update_xaxes(showline=True, zeroline=False, linecolor="#979A9A", gridcolor=self.chart_bg)
        fig_line.update_yaxes(showline=True, zeroline=False, linecolor="#979A9A", gridcolor=self.chart_bg)

        fig = self.generate_subchart(df_chart, "volume", "line", "yellow")
        fig_line.add_trace(fig, row=2, col=1)

        return df_daily, fig_line

    def generate_chart_2_full(self, para_combination, line_colour, settings):
        file_format = para_combination["file_format"]
        save_path = plotguy.generate_filepath(para_combination=para_combination)

        if file_format == "parquet":
            df_backtest = pd.read_parquet(save_path)
        else:
            df_backtest = pd.read_csv(save_path, index_col=0)

        df_csv = df_backtest.copy()
        df_chart = self.prepare_df_chart(df_csv)

        df_signal = df_chart.copy()
        df_signal = df_signal.dropna(subset=["action"])
        df_signal = df_signal.loc[df_signal["action"] != ""].reset_index()

        signal_open_date, signal_open_close = [], []
        signal_close_date, signal_close_close, signal_close_reason = [], [], []

        for i, row in df_signal.iterrows():
            if (row["action"] == "open") and ((i + 1) != len(df_signal)):
                signal_open_date.append(df_signal.iloc[i]["date"])
                signal_open_close.append(df_signal.iloc[i]["close"])
                signal_close_date.append(df_signal.iloc[i + 1]["date"])
                signal_close_close.append(df_signal.iloc[i + 1]["close"])
                signal_close_reason.append(df_signal.iloc[i + 1]["action"])
            else:
                signal_open_date.append(df_signal.iloc[i - 1]["date"])
                signal_open_close.append(df_signal.iloc[i - 1]["close"])
                signal_close_date.append(df_signal.iloc[i]["date"])
                signal_close_close.append(df_signal.iloc[i]["close"])
                signal_close_reason.append(df_signal.iloc[i]["action"])

        df_signal["signal_open_date"] = signal_open_date
        df_signal["signal_open_close"] = signal_open_close
        df_signal["signal_close_date"] = signal_close_date
        df_signal["signal_close_close"] = signal_close_close
        df_signal["close_reason"] = signal_close_reason
        df_signal = df_signal.set_index("date")

        open_date, open_close = [], []
        close_date, close_close, close_reason = [], [], []

        for _, row in df_chart.iterrows():
            try:
                open_date.append(df_signal.loc[row["date"]].signal_open_date.strftime("%Y-%m-%d"))
            except Exception:
                open_date.append(None)
            try:
                open_close.append(df_signal.loc[row["date"]].signal_open_close)
            except Exception:
                open_close.append(None)
            try:
                close_date.append(df_signal.loc[row["date"]].signal_close_date.strftime("%Y-%m-%d"))
            except Exception:
                close_date.append(None)
            try:
                close_close.append(df_signal.loc[row["date"]].signal_close_close)
            except Exception:
                close_close.append(None)
            try:
                close_reason.append(df_signal.loc[row["date"]].close_reason)
            except Exception:
                close_reason.append(None)

        df_signal_final = pd.DataFrame(
            {
                "open_date": open_date,
                "open_close": open_close,
                "close_date": close_date,
                "close_close": close_close,
                "close_reason": close_reason,
            }
        )
        df_signal_final["pctchange"] = ((df_signal_final["close_close"] - df_signal_final["open_close"]) / df_signal_final["open_close"]) * 100
        df_signal_final["open_close"] = df_signal_final["open_close"].round(2)
        df_signal_final["close_close"] = df_signal_final["close_close"].round(2)
        df_signal_final["pctchange"] = df_signal_final["pctchange"].round(2)

        hover_template = "<br>".join(
            [
                "Close Reason: %{customdata[4]}",
                "Open Date: %{customdata[0]}",
                "Close Date: %{customdata[2]}",
                "Open Close: %{customdata[1]}",
                "Close Close: %{customdata[3]}",
                "Pct Change: %{customdata[5]}%",
            ]
        )

        title = ""
        para_dict = para_combination["para_dict"]
        for key in para_dict:
            if len(title) > 100:
                title += "<br>"
            title += f"{key}:{para_combination[key]} "

        # Count subcharts
        subchart_count = 0
        for k in ("subchart_1", "subchart_2"):
            if k in settings:
                subchart_count += 1

        row_height = [1] if subchart_count == 0 else ([0.85, 0.15] if subchart_count == 1 else [0.7, 0.15, 0.15])

        fig_line = make_subplots(rows=subchart_count + 1, cols=1, row_heights=row_height, shared_xaxes=True)
        fig_line.update_layout(title={"text": title, "font": {"size": 12}, "y": 0.97})
        fig_line.update_xaxes(showline=True, zeroline=False, linecolor="white", gridcolor="rgba(0, 0, 0, 0)")
        fig_line.update_yaxes(showline=True, zeroline=False, linecolor="white", gridcolor="rgba(0, 0, 0, 0)")

        fig_line.add_trace(
            go.Scatter(
                mode="lines",
                hoverinfo="skip",
                x=df_chart["date"],
                y=df_chart["equity_value"],
                line=dict(color=line_colour, width=1),
                name="Strategy Equity",
            ),
            row=1,
            col=1,
        )

        df_close = pd.DataFrame({"close": df_backtest["close"]})
        fig_line.add_trace(
            go.Scatter(
                mode="lines",
                customdata=df_close,
                hovertemplate="Stock Price: %{customdata[0]}",
                x=df_chart["date"],
                y=df_chart["bah"],
                line=dict(color="Grey", width=1),
                name="",
            ),
            row=1,
            col=1,
        )

        fig_line.update_layout(
            plot_bgcolor=self.chart_bg,
            paper_bgcolor=self.chart_bg,
            height=500,
            margin=dict(l=85, r=25, t=35, b=20),
            font={"color": "white", "size": 9},
            yaxis={"title": "Equity"},
            xaxis={"title": ""},
        )

        fig_line.add_trace(
            go.Scatter(
                mode="markers",
                customdata=df_signal_final,
                hovertemplate=hover_template,
                x=df_chart["date"],
                y=df_chart["open"],
                marker=dict(color="rgba(0, 0, 0, 0)", size=9, line=dict(color="yellow", width=2.5)),
                name="open",
            ),
            row=1,
            col=1,
        )
        fig_line.add_trace(
            go.Scatter(
                mode="markers",
                customdata=df_signal_final,
                hovertemplate=hover_template,
                x=df_chart["date"],
                y=df_chart["close_logic"],
                visible="legendonly",
                marker=dict(color="rgba(0, 0, 0, 0)", size=9, line=dict(color="green", width=2.5)),
                name="close_logic",
            ),
            row=1,
            col=1,
        )
        fig_line.add_trace(
            go.Scatter(
                mode="markers",
                customdata=df_signal_final,
                hovertemplate=hover_template,
                x=df_chart["date"],
                y=df_chart["profit_target"],
                visible="legendonly",
                marker=dict(color="rgba(0, 0, 0, 0)", size=9, line=dict(color="red", width=2.5)),
                name="profit_target",
            ),
            row=1,
            col=1,
        )
        fig_line.add_trace(
            go.Scatter(
                mode="markers",
                customdata=df_signal_final,
                hovertemplate=hover_template,
                x=df_chart["date"],
                y=df_chart["stop_loss"],
                visible="legendonly",
                marker=dict(color="rgba(0, 0, 0, 0)", size=9, line=dict(color="Cyan", width=2.5)),
                name="stop_loss",
            ),
            row=1,
            col=1,
        )

        row_idx = 2
        try:
            fig = self.generate_subchart(df_chart, settings["subchart_1"][0], settings["subchart_1"][1], "yellow")
            fig_line.add_trace(fig, row=row_idx, col=1)
            row_idx += 1
        except Exception:
            pass

        try:
            fig = self.generate_subchart(df_chart, settings["subchart_2"][0], settings["subchart_2"][1], "#FF01FE")
            fig_line.add_trace(fig, row=row_idx, col=1)
        except Exception:
            pass

        return df_csv, fig_line

    def generate_subchart(self, df_chart: pd.DataFrame, element: str, line_type: str, line_color: str):
        if line_type == "bar":
            return go.Bar(
                hoverinfo="skip",
                x=df_chart["date"],
                y=df_chart[element],
                showlegend=True,
                marker_color=line_color,
                marker_line_color=line_color,
                name=element,
            )
        return go.Scatter(
            mode="lines",
            hoverinfo="skip",
            x=df_chart["date"],
            y=df_chart[element],
            showlegend=True,
            line=dict(color=line_color, width=1.5),
            name=element,
        )

    # -------------------------
    # Histogram + stats
    # -------------------------
    def generate_histogram(self, df: pd.DataFrame, period: int, mode: str):
        chart_bg = self.chart_bg
        col_pct = f"pct_change_{period}"
        col_rise = f"max_rise_{period}"
        col_fall = f"max_fall_{period}"

        df_his = df.copy()
        df_his[col_pct] = df_his["close"].pct_change(period).shift(-period) * 100
        df_his[col_pct] = df_his[col_pct].map(lambda x: round(x, 2))

        df_his[col_rise] = (df_his["high"].rolling(period).max().shift(-period) / df_his["close"] - 1) * 100
        df_his[col_fall] = (df_his["low"].rolling(period).min().shift(-period) / df_his["close"] - 1) * 100
        df_his[col_rise] = df_his[col_rise].map(lambda x: round(x, 2))
        df_his[col_fall] = df_his[col_fall].map(lambda x: round(x, 2))

        if mode == "backtest":
            df_his = df_his[df_his["action"] == "open"]
        else:
            df_his = df_his[df_his["logic"] == "trade_logic"]

        df_his = df_his[["date", col_pct, col_rise, col_fall]]

        margin = dict(l=5, r=5, t=5, b=15)
        h, w = 70, 120
        font = {"color": "white", "size": 8}

        def _base_fig():
            fig = go.Figure()
            fig.update_layout(plot_bgcolor=chart_bg, paper_bgcolor=chart_bg, height=h, width=w, margin=margin, font=font, bargap=0.1)
            fig.update_xaxes(showline=True, zeroline=False, linecolor="#979A9A", gridcolor="#1f2c56")
            fig.update_yaxes(showline=True, zeroline=False, linecolor="#979A9A", gridcolor="#626567")
            return fig

        fig_pct = _base_fig()
        fig_pct.add_trace(go.Histogram(x=df_his[col_pct], marker_color="Cyan"))

        fig_rise = _base_fig()
        fig_rise.add_trace(go.Histogram(x=df_his[col_rise], marker_color="Yellow"))

        fig_fall = _base_fig()
        fig_fall.add_trace(go.Histogram(x=df_his[col_fall], marker_color="Fuchsia"))

        return fig_pct, fig_rise, fig_fall

    def path_to_dict(self, path: str, file_format: str):
        filename = "file=" + path.split("file=")[-1].split(f".{file_format}")[0]
        parameters = {}

        for element in filename.split("&"):
            key, content = element.split("=")
            if key == "date":
                parameters["startdate"], parameters["enddate"] = content[:8], content[8:]
            elif key == "summary_mode":
                continue
            else:
                parameters[key] = content

        return parameters

    def generate_df_stat(self, df: pd.DataFrame, period: int):
        col_pct = f"pct_change_{period}"
        col_rise = f"max_rise_{period}"
        col_fall = f"max_fall_{period}"

        df_his = df.copy()
        df_his[col_pct] = df_his["close"].pct_change(period).shift(-period) * 100
        df_his[col_pct] = df_his[col_pct].map(lambda x: round(x, 2))

        df_his[col_rise] = (df_his["high"].rolling(period).max().shift(-period) / df_his["close"] - 1) * 100
        df_his[col_fall] = (df_his["low"].rolling(period).min().shift(-period) / df_his["close"] - 1) * 100

        df_his = df_his[df_his["logic"] == "trade_logic"]

        df_his[col_rise] = df_his[col_rise].map(lambda x: round(x, 2))
        df_his[col_fall] = df_his[col_fall].map(lambda x: round(x, 2))

        return (
            round(df_his[col_pct].mean(), 2),
            round(df_his[col_rise].mean(), 2),
            round(df_his[col_fall].mean(), 2),
        )

    # -------------------------
    # Aggregation
    # -------------------------
    def aggregate_df(self):
        folders = []
        for filename in os.listdir():
            if os.path.isfile(f"{filename}/saved_strategies.csv"):
                folders.append(filename)
        folders.sort()

        equity_curves = []
        for folder in folders:
            df_saved = pd.read_csv(f"{folder}/saved_strategies.csv")
            for _, row in df_saved.iterrows():
                equity_path = f"{folder}/{row['path']}"

                item = {}
                para_combination = ast.literal_eval(row["para_combination"])
                file_format = para_combination["file_format"]

                save_name = plotguy.filename_only(para_combination)
                ref_code = plotguy.path_reference_code(save_name)

                parameters = {}
                for element in save_name.split("&")[:-1]:
                    key, content = element.split("=")
                    if key == "date":
                        parameters["startdate"], parameters["enddate"] = content[:8], content[8:]
                    elif key == "summary_mode":
                        continue
                    else:
                        parameters[key] = content

                item["para_combination"] = para_combination
                item["folder"] = folder
                item["py"] = parameters["file"]
                item["path"] = equity_path
                item["parameters"] = parameters
                item["format"] = file_format
                item["ref"] = ref_code

                item["performance"] = {
                    "initial_capital": row["initial"],
                    "net_profit_to_mdd": row["net_profit_to_mdd"],
                    "net_profit": row["net_profit"],
                    "mdd_dollar": row["mdd_dollar"],
                    "mdd_pct": row["mdd_pct"],
                    "num_of_trade": row["num_of_trade"],
                    "num_of_win": row["num_of_win"],
                    "return_on_capital": row["return_on_capital"],
                    "total_commission": row["total_commission"],
                    "sharpe_ratio": row["sharpe"],
                }

                start_year = datetime.datetime.strptime(parameters["startdate"], "%Y%m%d").year
                end_year = datetime.datetime.strptime(parameters["enddate"], "%Y%m%d").year
                for year in range(start_year, end_year + 1):
                    item["performance"][f"{year}"] = row[f"{year}"]
                    item["performance"][f"{year}_win_count"] = row[f"{year}_win_count"]

                equity_curves.append(item)

        # Color assignment
        line_colour = []
        for c in range(len(equity_curves)):
            profile = c % 6
            degree = (c // 6) / math.ceil(len(equity_curves) / 6)
            line_colour.append(self.assign_colour(profile, degree))

        df = pd.DataFrame.from_dict(equity_curves, orient="columns")
        df["line_colour"] = line_colour
        df.index += 1

        choices = []
        for i in list(df.index):
            line_height = "120%"

            parameters_text = []
            for key, value in df.iloc[i - 1].parameters.items():
                if key != "file":
                    parameters_text.append(html.Div(f"{key} : {value}", style={"line-height": line_height}))

            performance_text = []
            perf = df.iloc[i - 1].performance

            if "initial_capital" in perf:
                performance_text.append(
                    html.Div(f"Initial Capital : {int(perf['initial_capital']):,}", style={"line-height": line_height})
                )
            if "num_of_trade" in perf:
                performance_text.append(
                    html.Div(f"Number of Trade : {int(perf['num_of_trade']):,}", style={"line-height": line_height})
                )
            if "mdd_pct" in perf:
                performance_text.append(
                    html.Div(f"MDD Percentage : {perf['mdd_pct']/100:.0%}", style={"line-height": line_height})
                )
            if "return_on_capital" in perf:
                performance_text.append(
                    html.Div(f"ReturnOnCapital : {perf['return_on_capital']/100:.0%}", style={"line-height": line_height})
                )

            choices.append(
                {
                    "label": html.Div(
                        [
                            html.Div("-----------------------------------------------------------------"),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col([html.Div("Curve", style={"line-height": line_height})], width=3),
                                                    dbc.Col([html.Div(str(i).zfill(3), style={"line-height": line_height})], width=6),
                                                ]
                                            ),
                                            html.Div(style={"height": "5px"}),
                                            dbc.Row(
                                                [
                                                    dbc.Col([html.Div("folder", style={"line-height": line_height})], width=3),
                                                    dbc.Col([html.Div(df.iloc[i - 1].folder, style={"line-height": line_height})], width=6),
                                                ]
                                            ),
                                            html.Div(style={"height": "5px"}),
                                            dbc.Row(
                                                [
                                                    dbc.Col([html.Div(".py", style={"line-height": line_height})], width=3),
                                                    dbc.Col([html.Div(df.iloc[i - 1].py, style={"line-height": line_height})], width=6),
                                                ]
                                            ),
                                            html.Div(style={"height": "5px"}),
                                            dbc.Row(
                                                [
                                                    dbc.Col([html.Div("parameters", style={"line-height": line_height})], width=3),
                                                    dbc.Col([html.Div(parameters_text, style={"line-height": line_height})], width=6),
                                                ]
                                            ),
                                            html.Div(style={"height": "5px"}),
                                            dbc.Row(
                                                [
                                                    dbc.Col([html.Div("performance", style={"line-height": line_height})], width=3),
                                                    dbc.Col([html.Div(performance_text, style={"line-height": line_height})], width=6),
                                                ]
                                            ),
                                            html.Div(style={"height": "5px"}),
                                            dbc.Row(
                                                [
                                                    dbc.Col([html.Div("format", style={"line-height": line_height})], width=3),
                                                    dbc.Col([html.Div(df.iloc[i - 1].format, style={"line-height": line_height})], width=6),
                                                ]
                                            ),
                                            html.Div(style={"height": "5px"}),
                                            dbc.Row(
                                                [
                                                    dbc.Col([html.Div("ref", style={"line-height": line_height})], width=3),
                                                    dbc.Col([html.Div(df.iloc[i - 1].ref, style={"line-height": line_height})], width=6),
                                                ]
                                            ),
                                            html.Div(style={"height": "5px"}),
                                        ],
                                        width=12,
                                    )
                                ]
                            ),
                        ],
                        style={"display": "inline-block", "width": "320px"},
                    ),
                    "value": i,
                }
            )

        checklist = html.Div(
            [
                dcc.Checklist(
                    options=choices,
                    id="curve_checklist",
                    value=list(df.index),
                    inputStyle={
                        "margin-left": "3px",
                        "margin-right": "10px",
                        "background-color": "red",
                        "vertical-align": "top",
                        "position": "relative",
                        "top": "1.6em",
                    },
                )
            ]
        )

        row = html.Div(
            dbc.Row(
                [
                    html.Div(checklist),
                    html.Div(style={"height": "5px"}),
                    html.Div("--------------------------------------------------------------------------------", style={"padding-left": "23px"}),
                    html.Div(style={"height": "10px"}),
                ]
            ),
            style={"padding": "0px", "font-size": "12px"},
        )

        return df, row

    def aggregate_performance(self, total_dict, year_list, risk_free_rate):
        num_of_trade = f"{int(total_dict['num_of_trade']):,}"
        net_profit = f"{int(round(total_dict['net_profit'], 0)):,}"
        net_profit_to_mdd = round(total_dict["net_profit_to_mdd"], 2)

        try:
            win_rate = "{:.0%}".format(float(total_dict["win_rate"]))
        except Exception:
            win_rate = "--"

        annualized_std = total_dict["annualized_std"]
        annualized_return = total_dict["annualized_return"]

        if isinstance(risk_free_rate, str):
            try:
                if risk_free_rate == "geometric_mean":
                    risk_free_rate_float = plotguy.get_geometric_mean_of_yearly_rate(year_list[0], year_list[-1])
                else:
                    risk_free_rate_float = plotguy.get_latest_fed_fund_rate()
            except Exception:
                risk_free_rate_float = 2
                print(f"Network error. Risk free rate: {risk_free_rate_float:.2f} %")
        else:
            risk_free_rate_float = risk_free_rate
            print(f"Risk free rate: {risk_free_rate_float:.2f} %")

        annualized_sr = (
            (annualized_return - float(risk_free_rate_float) / 100) / annualized_std if annualized_std > 0 else 0
        )

        total_commission = f"{int(total_dict['total_commission']):,}"
        return_on_capital = "{:.0%}".format(total_dict["return_on_capital"])
        annualized_return_fmt = "{:.0%}".format(total_dict["annualized_return"])
        annualized_std_fmt = "{:.0%}".format(total_dict["annualized_std"])
        annualized_sr = round(annualized_sr, 2)
        mdd_dollar = f"{int(total_dict['mdd_dollar']):,}"
        mdd_pct = "{:.0%}".format(total_dict["mdd_pct"])

        year_col, year_count_col, year_win_rate_col, year_return_col = [], [], [], []
        year_count = []
        year_return_list = []

        for year in year_list:
            year_col.append(html.Div(year))
            year_count.append(int(total_dict[f"{year}"]))
            year_count_col.append(html.Div(int(total_dict[f"{year}"])))

            try:
                rate = total_dict[f"{year}_win_count"] / total_dict[f"{year}"]
                year_win_rate_col.append(html.Div("{:.0%}".format(rate)))
            except Exception:
                year_win_rate_col.append(html.Div("--"))

            year_return = total_dict.get(f"{year}_return", 0)
            try:
                year_return_list.append(float(year_return))
                year_return_col.append(html.Div("{:.0%}".format(year_return)))
            except Exception:
                year_return_list.append(0)
                year_return_col.append(html.Div("--"))

        cov_count = round(np.std(year_count) / np.mean(year_count), 2) if np.mean(year_count) else 0
        cov_return = 0 if np.mean(year_return_list) == 0 else round(np.std(year_return_list) / np.mean(year_return_list), 2)

        return html.Div(
            [
                html.Div("Performance", style={"color": "Cyan", "font-size": "15px"}),
                html.Div(style={"height": "5px"}),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div("Number of Trade"),
                                html.Div("Net Profit"),
                                html.Div("Net Profit/MDD"),
                                html.Div("Win Rate"),
                                html.Div("COV (Count)"),
                                html.Div("COV (Return)"),
                                html.Div("Total Commission"),
                                html.Div("Risk Free Rate"),
                                html.Div(style={"height": "10px"}),
                                html.Div("Return on Capital"),
                                html.Div("Ann. Return"),
                                html.Div("Ann. Std"),
                                html.Div("Ann. Sharpe Ratio"),
                                html.Div("MDD Dollar"),
                                html.Div("MDD Percentage"),
                            ],
                            style={"padding-left": "13px"},
                            width=7,
                        ),
                        dbc.Col(
                            [
                                html.Div(num_of_trade),
                                html.Div(net_profit),
                                html.Div(net_profit_to_mdd),
                                html.Div(win_rate),
                                html.Div(cov_count),
                                html.Div(cov_return),
                                html.Div(total_commission),
                                html.Div("{:.2%}".format(risk_free_rate_float / 100)),
                                html.Div(style={"height": "10px"}),
                                html.Div(return_on_capital),
                                html.Div(annualized_return_fmt),
                                html.Div(annualized_std_fmt),
                                html.Div(annualized_sr),
                                html.Div(mdd_dollar),
                                html.Div(mdd_pct),
                            ],
                            style={"text-align": "center"},
                            width=5,
                        ),
                    ]
                ),
                html.Div(style={"height": "10px"}),
                html.Div("Performance by Year", style={"color": "Cyan", "font-size": "15px"}),
                html.Div(style={"height": "5px"}),
                dbc.Row(
                    [
                        dbc.Col(width=3),
                        dbc.Col("Count", style={"color": "Yellow", "font-size": "15px"}, width=3),
                        dbc.Col("WinRate", style={"color": "Yellow", "font-size": "15px"}, width=3),
                        dbc.Col("Return", style={"color": "Yellow", "font-size": "15px"}, width=3),
                    ],
                    style={"text-align": "center"},
                ),
                dbc.Row(
                    [
                        dbc.Col(year_col, width=3),
                        dbc.Col(year_count_col, width=3),
                        dbc.Col(year_win_rate_col, width=3),
                        dbc.Col(year_return_col, width=3),
                    ],
                    style={"text-align": "center"},
                ),
            ]
        )
