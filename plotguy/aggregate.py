from dash import Dash, dcc, html, Input, Output, State, ALL
import dash_daq as daq
from .components import *
import time

class Aggregate:

    chart_bg = '#1f2c56'
    valid_colour = 'DeepSkyBlue'
    df_equity_curves = pd.DataFrame()
    df_ready = False

    init = True

    def __new__(self, risk_free_rate):

        chart_bg = self.chart_bg
        components = Components(0)
        valid_colour = self.valid_colour

        empty_line_chart = components.empty_line_chart()

        self.df_equity_curves, checklist_div = components.aggregate_df()


        app = Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO], suppress_callback_exceptions=True)

        my_css_data = """body {
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
        innerHtmlText = "<style>%s</style>" % my_css_data


        app.layout = html.Div([

            dash_dangerously_set_inner_html.DangerouslySetInnerHTML(innerHtmlText),

            dbc.Row([
                # Left Column
                dbc.Col(html.Div([

                    html.Div(style={'height': '10px', }),

                    html.Div(children=html.Div([
                        html.Div(style={'height': '10px', }),

                        html.Div('Equity Curves', style={'color': 'DeepSkyBlue', 'font-size': '17px'}),

                        html.Div(style={'height': '10px', }),

                        html.Div(id='checklist-container', children=checklist_div,
                                 style={
                                        'maxHeight': '625px',
                                        'minHeight': '625px',
                                        'overflow-y': 'scroll',
                                        'overflow-x': 'hidden',
                                        }),

                    ],), style={'padding': '5px 5px','padding-left':'15px', 'border-radius': '5px',
                                'font-size': '13px','background-color': chart_bg},
                    ),

                ],), style={'padding': '0', 'padding-left': '5px',}, width=3),

                # Right Column
                dbc.Col(html.Div([



                    html.Div(style={'height': '15px', }),

                    html.Div([

                        html.Div(style={'height': '10px', }),

                        html.Div(html.Div('Aggregate', id='aggregate_string', style={'color': valid_colour}),
                                 style={'vertical-align': 'top','margin-left': '20px',
                                        'position': 'relative', 'top': '0.08em',
                                        'display': 'inline-block'}),

                        html.Div(daq.BooleanSwitch(on=True, color=valid_colour, id='aggregate_boolean',
                                                   style={'height': '5px', }),
                                 style={'margin-left': '10px', 'display': 'inline-block'}),

                        html.Div(html.Div('Normalized', id='normalized_string', style={'color': 'Grey'}),
                                 style={'margin-left': '15px', 'vertical-align': 'top',
                                        'position': 'relative', 'top': '0.08em',
                                        'display': 'inline-block'}),

                        html.Div(daq.BooleanSwitch(on=False, color=valid_colour, id='normalized_boolean',
                                                   style={'height': '5px', }),
                                 style={'margin-left': '10px', 'display': 'inline-block'}),
                    ]),

                    # html.Div(style={'height': '10px', }),

                    html.Div(children=html.Div([

                        dbc.Row([
                            # dbc.Col(html.Div(style={'height': '10px', }),

                            dbc.Col([

                                html.Div(style={'height': '15px', }),

                                html.Div(id='performance_div',
                                         style={'padding-left': '10px', 'font-size': '14px'}),

                                html.Div(style={'height': '15px', }),

                            ],style={'padding':'0px', 'padding-left': '28px',},width=3),

                            dbc.Col([

                                html.Div(style={'height': '5px', }),

                                html.Div(id='chart_area',
                                         children=dcc.Graph(id='line_chart',
                                                            figure=empty_line_chart)),

                                html.Div(style={'height': '20px', }),

                            ],style={'padding':'0px','padding-right': '35px',},width=9),
                        ]),

                    ]), style={'padding': '0px','border-radius': '5px',
                               # 'background-color': chart_bg
                               }),

                ]), style={'padding': '0', 'padding-left': '5px'}, width=9),

            ]),

        ], style={'width': '1500px', 'margin': 'auto', 'padding': '0px', 'color': 'white'})


        # For initial curve list
        @app.callback(
            Output('checklist-container', 'children'),
            Input('checklist-container', 'children'),
        )
        def init_checklist(checklist_container):
            self.df_ready = False
            self.df_equity_curves, checklist_div = components.aggregate_df()
            self.df_ready = True

            return checklist_div


        @app.callback(
            Output('line_chart', 'figure'),
            Output('normalized_string', 'style'),
            Output('aggregate_string', 'style'),
            Output('performance_div', 'children'),
            Input('curve_checklist', 'value'),
            Input('normalized_boolean','on'),
            Input('aggregate_boolean', 'on')
        )
        def aggregate_chart(curve_checklist,normalized_boolean, aggregate_boolean):

            if self.init:
                self.init = False
                fig_line = px.line()
                fig_line.update_layout(title={'text': ''})
                fig_line.update_xaxes(showline=True, zeroline=False, linecolor='white', gridcolor='rgba(0, 0, 0, 0)')
                fig_line.update_yaxes(showline=True, zeroline=False, linecolor='white', gridcolor='rgba(0, 0, 0, 0)')
                fig_line.update_layout(plot_bgcolor='#1a2245', paper_bgcolor='#1a2245', height=650,
                                       margin=dict(l=0, r=25, t=60, b=0),
                                       showlegend=True,
                                       font={"color": "white", 'size': 10.5}, yaxis={'title': ''},
                                       xaxis={'title': ''}
                                       )
                normalized_style = {}
                aggregate_style = {}
                aggregate_performance = html.Div()
                return fig_line, normalized_style, aggregate_style, aggregate_performance

            while not self.df_ready:
                time.sleep(0.2)

            curve_checklist.sort()
            valid_colour = self.valid_colour

            title_text = ''

            if aggregate_boolean:
                aggregate_style = {'color': valid_colour}
                title_text += 'Aggregate '
            else:
                aggregate_style = {'color': 'Grey'}

            if normalized_boolean:
                normalized_style = {'color': valid_colour}
                title_text += 'Normalized '
            else:
                normalized_style = {'color': 'Grey'}

            title_text += 'Equity Curves'


            df_curves_info = self.df_equity_curves.loc[curve_checklist]
            folders = list(df_curves_info['folder'])
            pys = list(df_curves_info['py'])
            paths = list(df_curves_info['path'])
            para_combination = list(df_curves_info['para_combination'])

            # Aggregate Equity
            dfs = []
            for i, path in enumerate(paths):
                file_format = para_combination[i]['file_format']
                intraday = para_combination[i]['intraday']
                summary = para_combination[i]['summary_mode']

                # df_csv = pd.read_csv(path, index_col='date')
                if file_format == 'parquet':
                    df_csv = pd.read_parquet(path)  # Daraframe that may not be daily
                else:
                    df_csv = pd.read_csv(path, index_col=0)

                df_csv['date'] = pd.to_datetime(df_csv['date'], format='%Y-%m-%d')
                df_csv = df_csv.set_index('date')

                # print(df_csv)

                if (intraday or summary):
                    df = plotguy.resample_summary_to_daily(para_combination=para_combination[i],folder=folders[i])
                else:
                    df = df_csv

                if 'date' in df.columns:
                    df = df.set_index('date')

                df_current = pd.DataFrame()
                if normalized_boolean:
                    df_current[f'{folders[i]}_{pys[i]}_{i}'] = df['equity_value'] / (df['equity_value'].iloc[0] / 100000)
                else:
                    df_current[f'{folders[i]}_{pys[i]}_{i}'] = df['equity_value']
                df_current.index = df.index
                dfs.append(df_current.copy())

            df_all_curves = pd.concat(dfs, axis=1).sort_index()
            df_all_curves = df_all_curves.fillna(method='pad')
            df_all_curves = df_all_curves.fillna(method='backfill')
            df_all_curves["Aggregate_Equity"] = df_all_curves.sum(axis=1)
            if normalized_boolean:
                df_all_curves["Aggregate_Equity"] = df_all_curves["Aggregate_Equity"] / (len(df_all_curves.columns) - 1)
            # print(df_all_curves)

            # Performance
            dicts = []
            for i, row in df_curves_info.iterrows():
                d = row['performance'].copy()

                if normalized_boolean:
                    initial_capital = d['initial_capital']
                    d['initial_capital'] = d['initial_capital'] / (initial_capital / 100000)
                    d['net_profit'] = d['net_profit'] / (initial_capital / 100000)
                    d['total_commission'] = d['total_commission'] / (initial_capital / 100000)
                d.pop('net_profit_to_mdd')
                d.pop('mdd_dollar')
                d.pop('mdd_pct')
                d.pop('return_on_capital')
                d.pop('sharpe_ratio')
                dicts.append(d.copy())

            df_performance = pd.DataFrame.from_dict(dicts)

            total_dict = {}
            for element in df_performance.columns:
                total_dict[element] = df_performance[element].sum()


            total_dict['return_on_capital'] = total_dict['net_profit']/total_dict['initial_capital']
            total_dict['win_rate'] = total_dict['num_of_win']/total_dict['num_of_trade']



            df_agg = df_all_curves['Aggregate_Equity']
            dds = []
            dd_pct = []
            for i in range(len(df_agg)):
                max_list = list(df_agg[:i + 1]).copy()
                dd = max(max_list) - df_agg.iloc[i]
                dds.append(dd)
                dd_pct.append( dd / max(max_list))
                # print(dd, max(max_list), dd / max(max_list))
            mdd = max(dds)
            mdd_pct = max(dd_pct)

            total_dict['mdd_dollar'] = mdd
            total_dict['mdd_pct'] = mdd_pct
            total_dict['net_profit_to_mdd'] = total_dict['net_profit'] / total_dict['mdd_dollar']

            df_all_curves['date'] = pd.to_datetime(df_all_curves.index, format='%Y-%m-%d')
            holding_period_day = (df_all_curves.loc[df_all_curves.index[-1], 'date'] - df_all_curves.loc[
                df_all_curves.index[0], 'date']).days

            equity_value_pct_series = df_all_curves["Aggregate_Equity"].pct_change()
            equity_value_pct_series = equity_value_pct_series.dropna()

            return_on_capital = total_dict['return_on_capital']

            annualized_return = (1 + return_on_capital) ** (365 / holding_period_day) - 1

            annualized_std = equity_value_pct_series.std() * math.sqrt(365)

            if annualized_std > 0:
                annualized_sr = annualized_return / annualized_std
            else:
                annualized_sr = 0

            total_dict['annualized_return'] = annualized_return
            total_dict['annualized_std'] = annualized_std
            total_dict['annualized_sr'] = annualized_sr

            start_date_year = df_all_curves.loc[df_all_curves.index[0], 'date'].year
            end_date_year = df_all_curves.loc[df_all_curves.index[-1], 'date'].year

            year_list = list(range(start_date_year, end_date_year + 1))


            # Performance by year
            df_year = pd.DataFrame()
            df_year['equity_value'] = df_all_curves['Aggregate_Equity'].copy()
            df_year['date'] = df_agg.index.copy()
            df_year['date'] = pd.to_datetime(df_year['date'], format='%Y-%m-%d')
            df_year['year'] = pd.DatetimeIndex(df_year['date']).year

            first_equity_value = 0
            for year in year_list:
                if first_equity_value == 0:
                    first_equity_value = df_year.loc[df_year['year'] == year].iloc[0].equity_value
                last_equity_value = df_year.loc[df_year['year'] == year].iloc[-1].equity_value
                yearly_return = (last_equity_value - first_equity_value) / first_equity_value
                total_dict[f'{year}_return'] = yearly_return
                first_equity_value = last_equity_value

            aggregate_performance = components.aggregate_performance(total_dict, year_list, risk_free_rate)


            # Generate Chart
            if aggregate_boolean:
                fig_line = px.line()
                fig_line.update_layout(title={'text': title_text })
                fig_line.update_xaxes(showline=True, zeroline=False, linecolor='white', gridcolor='rgba(0, 0, 0, 0)')
                fig_line.update_yaxes(showline=True, zeroline=False, linecolor='white', gridcolor='rgba(0, 0, 0, 0)')
                fig_line.update_layout(plot_bgcolor='#1a2245', paper_bgcolor='#1a2245', height=650,
                                       margin=dict(l=0, r=25, t=60, b=0),
                                       showlegend=True,
                                       font={"color": "white", 'size': 10.5}, yaxis={'title': ''},
                                       xaxis={'title': ''}
                                       )
                fig_line.add_trace(go.Scatter(mode='lines', # hovertemplate=hovertemplate,
                                              x=df_all_curves.index, y=df_all_curves['Aggregate_Equity'],
                                              line=dict(color=valid_colour, width=1.5), name='Aggregate'), )
            else:
                fig_line = px.line()
                fig_line.update_layout(title={'text': title_text})
                fig_line.update_xaxes(showline=True, zeroline=False, linecolor='white', gridcolor='rgba(0, 0, 0, 0)')
                fig_line.update_yaxes(showline=True, zeroline=False, linecolor='white', gridcolor='rgba(0, 0, 0, 0)')
                fig_line.update_layout(plot_bgcolor='#1a2245', paper_bgcolor='#1a2245', height=650,
                                       margin=dict(l=0, r=25, t=60, b=0),
                                       showlegend=True,
                                       font={"color": "white", 'size': 10.5}, yaxis={'title': ''},
                                       xaxis={'title': ''},
                                       )
                # print(list(df_all_curves.columns))

                columns = list(df_all_curves.columns)[:-1]
                for curve_number, column in zip(curve_checklist,columns):
                    fig_line.add_trace(go.Scatter(mode='lines',  # hovertemplate=hovertemplate,
                                                  x=df_all_curves.index, y=df_all_curves[column],
                                                  line=dict(color=self.df_equity_curves.loc[curve_number].line_colour, width=1.5), name=f'Curve {str(curve_number).zfill(3)}'), )



            return fig_line, normalized_style, aggregate_style, aggregate_performance





        return app