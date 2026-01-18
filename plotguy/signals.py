from dash import Dash, dcc, html, Input, Output, State, ALL
from .components import *


class Signals:
    # Settings
    chart_bg = "#1f2c56"

    def __new__(self, all_para_combination, generate_filepath, settings):
        print(settings)

        chart_bg = self.chart_bg
        components = Components(0)
        empty_line_chart = components.empty_line_chart()
        para_dict = all_para_combination[0]["para_dict"]

        # Build a list of parameter combinations in a stable order (matching para_dict keys)
        para_combination_list = []
        for para in all_para_combination:
            para_values = []
            for key in para_dict:
                para_values.append(para[key])
            para_combination_list.append(para_values)

        radioitems_div = components.generate_radioitems(para_dict)
        period = settings["histogram_period"]

        stat_placeholder = ["---"] * 5
        para_placeholder = ["-----" for _ in para_dict]

        app = Dash(
            __name__,
            external_stylesheets=[dbc.themes.SUPERHERO],
            suppress_callback_exceptions=True,
        )

        my_css_data = """
        body {
          background-color: #1a2245;
        }

        /* width */
        ::-webkit-scrollbar {
          width: 10px !important;
          display: block !important;
        }

        /* Track */
        ::-webkit-scrollbar-track {
          background: #1f2c56 !important;
          border-radius: 10px !important;
          display: block !important;
        }

        /* Handle */
        ::-webkit-scrollbar-thumb {
          background: #154360;
          border-radius: 10px;
        }
        """
        inner_html_text = f"<style>{my_css_data}</style>"

        app.layout = html.Div(
            [
                dash_dangerously_set_inner_html.DangerouslySetInnerHTML(inner_html_text),
                html.Div(style={"height": "5px"}),
                html.Div(
                    dbc.Row(
                        [
                            # Left Column
                            dbc.Col(
                                html.Div(
                                    [
                                        html.Div(style={"height": "10px"}),
                                        html.Div(
                                            "Selected Parameters",
                                            style={"color": "Cyan", "font-size": "16px"},
                                        ),
                                        html.Div(
                                            id="title_area",
                                            children=components.selection_title(
                                                para_dict, para_placeholder
                                            ),
                                            style={
                                                "padding-left": "10px",
                                                "font-size": "14px",
                                            },
                                        ),
                                        html.Div(style={"height": "20px"}),
                                        html.Div(
                                            id="radioitems_container",
                                            children=radioitems_div,
                                            style={"padding-right": "30px"},
                                        ),
                                    ],
                                    style={
                                        "padding": "0px",
                                        "border-radius": "5px",
                                        "padding-left": "50px",
                                        "padding-right": "15px",
                                        "height": "704px",
                                        "background-color": "rgba(0, 0, 0, 0)",
                                    },
                                ),
                                style={"padding": "0"},
                                width=4,
                            ),
                            # Right Column
                            dbc.Col(
                                html.Div(
                                    [
                                        html.Div(style={"height": "10px"}),
                                        html.Div(
                                            [
                                                html.Div(
                                                    html.Img(),
                                                    style={"height": "5px"},
                                                ),
                                                html.Div(
                                                    id="chart_area",
                                                    children=dcc.Graph(
                                                        id="line_chart",
                                                        figure=empty_line_chart,
                                                    ),
                                                ),
                                            ],
                                            style={
                                                "padding": "5px",
                                                "border-radius": "5px",
                                                "padding-right": "25px",
                                                "background-color": chart_bg,
                                            },
                                        ),
                                        html.Div(style={"height": "5px"}),
                                        html.Div(
                                            children=html.Div(
                                                [
                                                    html.Div(style={"height": "5px"}),
                                                    html.Div(
                                                        "Average",
                                                        style={
                                                            "color": "Cyan",
                                                            "font-size": "14px",
                                                        },
                                                    ),
                                                    html.Div(
                                                        id="stat_area",
                                                        children=components.update_stat_div(
                                                            period,
                                                            stat_placeholder,
                                                            stat_placeholder,
                                                            stat_placeholder,
                                                        ),
                                                        style={
                                                            "padding": "0px 20px",
                                                            "text-align": "center",
                                                            "font-size": "12px",
                                                        },
                                                    ),
                                                    html.Div(style={"height": "5px"}),
                                                    html.Div(
                                                        "Histogram",
                                                        style={
                                                            "color": "Cyan",
                                                            "font-size": "14px",
                                                        },
                                                    ),
                                                    html.Div(
                                                        id="hist_area",
                                                        children=html.Div(),
                                                        style={
                                                            "padding": "0px 20px",
                                                            "text-align": "center",
                                                        },
                                                    ),
                                                ]
                                            ),
                                            style={
                                                "padding": "5px 25px",
                                                "border-radius": "5px",
                                                "height": "380px",
                                                "font-size": "13px",
                                                "background-color": chart_bg,
                                            },
                                        ),
                                    ]
                                ),
                                style={"padding": "0", "padding-left": "5px"},
                                width=8,
                            ),
                        ]
                    )
                ),
            ],
            style={
                "width": "1200px",
                "margin": "auto",
                "padding": "0px",
                "color": "white",
            },
        )

        @app.callback(
            Output("title_area", "children"),
            Output("stat_area", "children"),
            Output("line_chart", "figure"),
            Output("hist_area", "children"),
            State("title_area", "children"),
            State("stat_area", "children"),
            State("line_chart", "figure"),
            State("hist_area", "children"),
            Input({"type": "para_radioitems", "index": ALL}, "value"),
        )
        def display_output(title_area, stat_area, line_chart, hist_area, para_radioitems):
            # Incomplete selection
            if None in para_radioitems:
                return title_area, stat_area, line_chart, hist_area

            index = para_combination_list.index(para_radioitems)
            file_path = generate_filepath(para_combination=all_para_combination[index])

            file_format = all_para_combination[index]["file_format"]
            if file_format == "csv":
                df = pd.read_csv(file_path)
            elif file_format == "parquet":
                df = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")

            # Count subcharts
            subchart_count = 0
            if settings.get("subchart_1"):
                subchart_count += 1
            if settings.get("subchart_2"):
                subchart_count += 1

            if subchart_count == 0:
                row_height = [1]
                height = 350
            elif subchart_count == 1:
                row_height = [0.85, 0.15]
                height = 420
            else:
                row_height = [0.7, 0.15, 0.15]
                height = 500

            title = " ".join(
                f"{key}:{para_radioitems[i]}" for i, key in enumerate(para_dict)
            )

            # Close chart + signal markers
            df_chart = df.copy()
            df_chart["open_signal"] = [
                row["close"] if row.get("logic") == "trade_logic" else None
                for _, row in df_chart.iterrows()
            ]

            fig_line = make_subplots(
                rows=subchart_count + 1,
                cols=1,
                row_heights=row_height,
                shared_xaxes=True,
            )
            fig_line.update_layout(title={"text": title, "font": {"size": 12}})
            fig_line.update_xaxes(
                showline=True,
                zeroline=False,
                linecolor="white",
                gridcolor="rgba(0, 0, 0, 0)",
            )
            fig_line.update_yaxes(
                showline=True,
                zeroline=False,
                linecolor="white",
                gridcolor="rgba(0, 0, 0, 0)",
            )
            fig_line.update_layout(
                plot_bgcolor=self.chart_bg,
                paper_bgcolor=self.chart_bg,
                height=height,
                margin=dict(l=85, r=80, t=35, b=20),
                font={"color": "white", "size": 9},
                yaxis={"title": "Close"},
                xaxis={"title": ""},
            )

            fig_line.add_trace(
                go.Scatter(
                    mode="lines",
                    hoverinfo="skip",
                    x=df_chart["date"],
                    y=df_chart["close"],
                    line=dict(color="#00FFFF", width=1),
                    name="Close",
                ),
                row=1,
                col=1,
            )
            fig_line.add_trace(
                go.Scatter(
                    mode="markers",
                    x=df_chart["date"],
                    y=df_chart["open_signal"],
                    marker=dict(
                        color="rgba(0, 0, 0, 0)",
                        size=9,
                        line=dict(color="yellow", width=2.5),
                    ),
                    name="Signal",
                ),
                row=1,
                col=1,
            )

            # Add subcharts (if configured)
            next_row = 2
            try:
                if settings.get("subchart_1"):
                    fig = components.generate_subchart(
                        df_chart,
                        settings["subchart_1"][0],
                        settings["subchart_1"][1],
                        "yellow",
                    )
                    fig_line.add_trace(fig, row=next_row, col=1)
                    next_row += 1
            except Exception:
                pass

            try:
                if settings.get("subchart_2"):
                    fig = components.generate_subchart(
                        df_chart,
                        settings["subchart_2"][0],
                        settings["subchart_2"][1],
                        "#FF01FE",
                    )
                    fig_line.add_trace(fig, row=next_row, col=1)
            except Exception:
                pass

            line_chart = fig_line

            # Histogram area
            title_cols = [dbc.Col(width=2)]
            pct_cols = [dbc.Col(html.Div("pct_change", style={"font-size": "12px"}), width=2)]
            rise_cols = [dbc.Col(html.Div("max_rise", style={"font-size": "12px"}), width=2)]
            fall_cols = [dbc.Col(html.Div("max_fall", style={"font-size": "12px"}), width=2)]

            pct_mean, rise_mean, fall_mean = [], [], []

            for p in period:
                title_cols.append(
                    dbc.Col(
                        f"{p} Days",
                        style={"font-size": "12px", "text-align": "center"},
                        width=2,
                    )
                )

                fig_pct, fig_rise, fig_fall = components.generate_histogram(df, p, "signal")
                pct_cols.append(
                    dbc.Col(
                        dcc.Graph(figure=fig_pct, config={"displayModeBar": False}),
                        style={"padding": "0"},
                        width=2,
                    )
                )
                rise_cols.append(
                    dbc.Col(
                        dcc.Graph(figure=fig_rise, config={"displayModeBar": False}),
                        style={"padding": "0"},
                        width=2,
                    )
                )
                fall_cols.append(
                    dbc.Col(
                        dcc.Graph(figure=fig_fall, config={"displayModeBar": False}),
                        style={"padding": "0"},
                        width=2,
                    )
                )

                stat_pct, stat_rise, stat_fall = components.generate_df_stat(df, p)
                pct_mean.append(stat_pct)
                rise_mean.append(stat_rise)
                fall_mean.append(stat_fall)

            hist_area = [
                html.Div(dbc.Row(title_cols)),
                html.Div(style={"height": "5px"}),
                html.Div(dbc.Row(pct_cols)),
                html.Div(style={"height": "5px"}),
                html.Div(dbc.Row(rise_cols)),
                html.Div(dbc.Row(fall_cols)),
            ]

            title_area = components.selection_title(para_dict, para_radioitems)
            stat_area = components.update_stat_div(period, pct_mean, rise_mean, fall_mean)

            return title_area, stat_area, line_chart, hist_area

        return app
