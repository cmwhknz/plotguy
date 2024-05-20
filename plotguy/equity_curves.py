from dash import Dash, dcc, html, Input, Output, State, ALL
from .components import *

import os

class Plot:
    # Settings
    chart_bg = '#1f2c56'

    # State
    graph_df = pd.DataFrame()
    sort_method_current = ''
    filter_list = []
    add_button_count = 0
    chart1_button_clicks = 0
    chart2_button_clicks = 0
    chart3_button_clicks = 0
    chart3_back_button_clicks = 0
    save_clicks = 0
    line_selected = -1
    button_list = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    chart_type = 1
    relayoutData = []
    init = True
    select_all_status = False
    save_path = ''
    initial_capital = 0
    return_in_capital = 0
    mdd = 0
    sharpe = 0

    start_date_datetime = 0
    end_date_datetime = 0

    para_combination = {}
    save_path = 'save_path'
    net_profit_to_mdd = 0
    net_profit = 0
    mdd_dollar = 0
    mdd_pct = 0
    num_of_trade = 0
    num_of_win = 0
    return_on_capital = 0
    sharpe = 0
    total_commission = 0
    year_list = []
    year_count = []
    year_win_count = []


    def __new__(self, all_para_combination, result_df, settings, number_of_curves):
        # start_date, end_date and para_dict is the same for all combination, ie use the 1st one
        start_date = all_para_combination[0]['start_date']
        end_date = all_para_combination[0]['end_date']
        para_dict = all_para_combination[0]['para_dict']
        intraday = all_para_combination[0]['intraday']
        summary_mode = all_para_combination[0]['summary_mode']
        freq = all_para_combination[0]['freq']
        # risk_free_rate = all_para_combination[0]['risk_free_rate']
        self.initial_capital = all_para_combination[0]['sec_profile']['initial_capital']

        # For chart3 initial date range
        self.start_date_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date_datetime = self.start_date_datetime + datetime.timedelta(days=10)



        chart_bg = self.chart_bg

        components = Components(number_of_curves)

        start_date_year = datetime.datetime.strptime(start_date, '%Y-%m-%d').year
        end_date_year = datetime.datetime.strptime(end_date, '%Y-%m-%d').year
        self.year_list = list(range(start_date_year, end_date_year + 1))

        empty_line_chart = components.empty_line_chart()
        checkbox_div, code_list = components.update_checkbox_div(para_dict, result_df)
        col_df = result_df.columns.values.tolist()
        df_col = pd.DataFrame([['-----' for column in col_df]], columns=col_df)
        performance_matrix = components.update_performance_matrix(start_date, end_date, df_col.iloc[0].copy(), para_dict, '-----', '-----')
        filter_div = components.update_filter_div([])


        sort_method_dropdown  = components.sort_method_dropdown
        filter_dropdown = components.filter_dropdown
        filter_dropdown_disabled = components.filter_dropdown_disabled
        filter_input = components.filter_input
        filter_input_disabled = components.filter_input_disabled
        add_button_style = components.add_button_style
        add_button_style_disabled = components.add_button_style_disabled



        app = Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO], suppress_callback_exceptions=True)

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
        innerHtmlText = "<style>%s</style>" % my_css_data

        app.layout = html.Div([

            dash_dangerously_set_inner_html.DangerouslySetInnerHTML(innerHtmlText),


            html.Div(style={'height': '10px', }),
            html.Div(
                dbc.Row([
                    # Left Column
                    dbc.Col(html.Div([
                        html.Div(style={'height': '15px', }),
                        html.Div('Sorting Method', style={'color': 'Cyan', 'margin-left': '15px', 'font-size': '15px'}),
                        html.Div(style={'height': '10px', }),
                        sort_method_dropdown,
                        html.Div(style={'height': '15px', }),
                        html.Div('Parameters', style={'color': 'Cyan', 'margin-left': '15px', 'font-size': '15px'}),
                        html.Div(id='checklist-container',children=checkbox_div),
                        html.Div(style={'height': '10px', }),
                        html.Div('Filters for', style={'color': 'Cyan', 'margin-left': '15px', 'font-size': '15px'}),
                        html.Div(style={'height': '10px', }),
                        html.Div(id='filter', children=filter_div,style={'padding': '0px 20px',
                                                                         'padding-right': '10px',
                                                                         'font-size': '12px'}),
                        dbc.Row([
                            dbc.Col(id='filter_dropdown_div',children=filter_dropdown,width=8),
                            dbc.Col(id='filter_input_div',children=filter_input,width=3),
                        ]),
                        html.Div(style={'height': '10px', }),
                        html.Div(id='add_button',children=html.Div([html.Div(style={'height': '5px'}),'Add Filter']),
                                 style=add_button_style),

                    ],style={'padding':'0px','border-radius':'5px','height': '704px',
                             'background-color':'rgba(0, 0, 0, 0)'})
                        ,style={'padding':'0','padding-left':'5px'}, width=2),

                    # Middle Column
                    dbc.Col(html.Div([

                        html.Div(id='performance_matrix', children=performance_matrix),

                    ], style={'padding': '5px',})
                        , style={'padding': '0', }, width=2),


                    # Right Column
                    dbc.Col(html.Div([
                        html.Div(style={'height': '5px', }),

                        html.Div(id='main_chart_area',
                            children=[html.Div(html.Img(),style={'height':'10px'}),

                                      dbc.Row([
                                          dbc.Col(html.Div(id='chart1_button', children='',
                                                           style={'text-align': 'left', 'cursor': 'pointer',
                                                                  'font-size': '14px'}), width=4),
                                          dbc.Col(html.Div(html.Img()), width=4),
                                          dbc.Col(html.Div(id='chart2_button', children=''), width=4),
                                      ], style={'margin': '0px 5px'}),
                                      html.Div(id='chart_area',
                                               children=dcc.Graph(id='line_chart', figure=empty_line_chart)),

                                      html.Div(style={'height': '10px', }),

                                     ],style={'padding':'5px','border-radius':'5px','background-color':chart_bg}),


                        html.Div(id='chart3_area',
                                 children=[html.Div(html.Img(), style={'height': '10px'}),

                                           dbc.Row([
                                               dbc.Col(html.Div(id='chart3_back_button', children='< Back',
                                                                style={'text-align': 'left', 'cursor': 'pointer',
                                                                       'font-size': '14px'}), width=4),
                                               dbc.Col(html.Div(html.Img()), width=4)

                                           ], style={'margin': '0px 5px'}),
                                           html.Div(id='chart3',
                                                    children=dcc.Graph(id='chart3_chart', figure=empty_line_chart)),

                                           html.Div(style={'height': '10px', }),

                                           ],
                                 style={'padding': '5px', 'border-radius': '5px', 'background-color': chart_bg, 'display': 'none'}),

                        html.Div(id='chart3_dashboard',
                            children=[html.Div(style={'height': '5px', }),
                                  html.Div(children=dbc.Row([
                                                dbc.Col(width=1),
                                                dbc.Col(children=html.Div([
                                                    html.Div(style={'height': '0.5px',}),
                                                    dcc.DatePickerRange(
                                                        id='chart3_date',
                                                        style={'margin-left':'15px'},
                                                        month_format='MMM Do, YY',
                                                        end_date_placeholder_text='MMM Do, YY',
                                                        start_date=datetime.date(self.start_date_datetime.year, self.start_date_datetime.month, self.start_date_datetime.day),
                                                        end_date=datetime.date(self.end_date_datetime.year, self.end_date_datetime.month, self.end_date_datetime.day),)
                                                ]),width=6),
                                                dbc.Col(children=html.Div(id='chart3_button',
                                                                          children=html.Div([html.Div(style={'height': '11px'}),'Show Intra Day Candlestick Chart']),
                                                                          style={'margin': '0px', 'width': '300px', 'backgroundColor': 'blue', 'padding': '0px',
                                                                                    'border-radius': '2.5px', 'text-align': 'center', 'cursor': 'pointer',
                                                                                    'font-size': '18px', 'height':'48px'}
                                                                    ), width=3),
                                            ], style={'width':'650px'})
                                           ,style={'padding': '15px 5px', 'border-radius': '5px', 'background-color': chart_bg})
                                  ], style={'display': 'none'}),

                        html.Div(style={'height': '5px', }),

                        html.Div(id='hist_area', children=html.Div(),
                                 style={'display': 'none'}),

                    ]), style={'padding':'0'}, width=8),
                ])
            ),

        ], style={'width':'1500px','margin':'auto','padding':'0px 10px','color':'white'})


        @app.callback(
            Output('line_chart', 'figure'),
            Output('filter', 'children'),
            Output('filter_dropdown_div', 'children'),
            Output('filter_input_div', 'children'),
            Output('add_button', 'style'),
            Output('filter_input', 'value'),
            Output('chart1_button', 'children'),
            Output('chart2_button', 'children'),
            Output('hist_area', 'children'),
            Output('hist_area', 'style'),
            Output('main_chart_area', 'style'),
            Output('chart3_dashboard', 'style'),
            Output('chart3_area', 'style'),
            Output('chart3_date', 'start_date'),
            Output('chart3_date', 'end_date'),
            Output('chart3_chart', 'figure'),
            Input('sort_method', 'value'),
            Input({'type': 'para-checklist', 'index': ALL}, 'value'),
            State('filter_dropdown_div', 'children'),
            State('filter_input_div', 'children'),
            State('line_chart', 'figure'),
            State('filter', 'children'),
            Input('add_button', 'n_clicks'),
            Input('chart1_button', 'n_clicks'),
            Input('chart2_button', 'n_clicks'),
            Input('chart3_button', 'n_clicks'),
            Input('chart3_back_button', 'n_clicks'),
            State('add_button', 'style'),
            State('filter_name', 'value'),
            State('filter_input', 'value'),
            State('hist_area', 'children'),
            State('hist_area', 'style'),
            State('main_chart_area', 'style'),
            State('chart3_dashboard', 'style'),
            State('chart3_area', 'style'),
            State('chart3_date', 'start_date'),
            State('chart3_date', 'end_date'),
            State('chart3_chart', 'figure'),
            [Input('button_' + str(i), 'n_clicks') for i in range(10)],
        )
        def display_output(sort_method,para_checklist,filter_dropdown_div,filter_input_div,fig_line,\
                           filter_div,add_button_clicks, \
                           chart1_button_clicks,chart2_button_clicks, chart3_button_clicks, chart3_back_button_clicks, \
                           _add_button_style,filter_name,filter_input_value,\
                           hist_area,hist_style,main_chart_area_style,chart3_dashboard_style, chart3_area_style, \
                           chart3_start_str, chart3_end_str, chart3_chart, *vals):

            chart1_button_text = ''
            chart2_button_text = ''

            chart1_button_click = False

            if chart3_button_clicks:
                if chart3_button_clicks > self.chart3_button_clicks:
                    self.chart3_button_clicks = chart3_button_clicks

                    if (chart3_start_str and chart3_end_str):
                        chart3_start = datetime.datetime.strptime(chart3_start_str, '%Y-%m-%d')
                        chart3_end = datetime.datetime.strptime(chart3_end_str, '%Y-%m-%d')
                        detla = chart3_end - chart3_start

                        if (detla.days > 60) or (detla.days < 1):
                            print('CandleStick Chart Limit: 1 to 60 Days')
                        else:
                            para_combination = self.para_combination
                            hist_style = {'display': 'none'}
                            file_format = para_combination['file_format']
                            chart3_chart = components.generate_chart_3(para_combination, chart3_start, chart3_end, freq, file_format)

                            main_chart_area_style = {'padding': '5px', 'border-radius': '5px', 'background-color': chart_bg, 'display': 'none'}
                            chart3_area_style = {'padding': '5px', 'border-radius': '5px', 'background-color': chart_bg,}


                    return fig_line, filter_div, filter_dropdown_div, filter_input_div, _add_button_style, \
                           None, chart1_button_text, chart2_button_text, hist_area, hist_style, \
                           main_chart_area_style, chart3_dashboard_style, chart3_area_style, \
                           chart3_start_str, chart3_end_str, chart3_chart

            if chart3_back_button_clicks:
                if chart3_back_button_clicks > self.chart3_back_button_clicks:
                    self.chart3_back_button_clicks = chart3_back_button_clicks
                    chart1_button_text = '< Back'
                    hist_style = {'padding': '5px', 'border-radius': '5px', 'background-color': chart_bg}
                    main_chart_area_style = {'padding': '5px', 'border-radius': '5px', 'background-color': chart_bg,}
                    chart3_area_style = {'padding': '5px', 'border-radius': '5px', 'background-color': chart_bg,
                                         'display': 'none'}


                    return fig_line, filter_div, filter_dropdown_div, filter_input_div, _add_button_style, \
                           None, chart1_button_text, chart2_button_text, hist_area, hist_style, \
                           main_chart_area_style, chart3_dashboard_style, chart3_area_style, \
                           chart3_start_str, chart3_end_str, chart3_chart


            if chart1_button_clicks:
                if chart1_button_clicks > self.chart1_button_clicks:
                    chart1_button_click = True
                    self.chart1_button_clicks = chart1_button_clicks
                    self. chart_type = 1
                    hist_style = {'display': 'none'}
                    chart3_dashboard_style = {'display': 'none'}

            if chart2_button_clicks:
                if chart2_button_clicks > self.chart2_button_clicks:
                    self.chart2_button_clicks = chart2_button_clicks
                    if self.chart_type == 2:
                        chart1_button_text = '< Back'
                        return fig_line, filter_div, filter_dropdown_div, filter_input_div, _add_button_style, \
                               None, chart1_button_text, chart2_button_text, hist_area, hist_style, \
                               main_chart_area_style, chart3_dashboard_style, chart3_area_style, \
                               chart3_start_str, chart3_end_str, chart3_chart
                    else:
                        if self.line_selected > -1:
                            para_combination = self.para_combination
                            line_colour = self.graph_df.iloc[self.line_selected].line_colour

                            if (intraday or summary_mode):
                                df, fig_line = components.generate_chart_2_summary(para_combination, line_colour)

                            else:
                                df, fig_line = components.generate_chart_2_full(para_combination, line_colour, settings)

                            self.chart_type = 2
                            chart1_button_text = '< Back'


                            # For Chart 3 date range
                            if intraday:
                                chart3_dashboard_style = {}

                                df_chart3 = df.loc[df['action'].isnull() == False].copy()
                                self.start_date_datetime = df_chart3.iloc[0].date - datetime.timedelta(days=1)
                                self.end_date_datetime = self.start_date_datetime + datetime.timedelta(days=10)

                                chart3_start_str = datetime.date(self.start_date_datetime.year, self.start_date_datetime.month,
                                              self.start_date_datetime.day)
                                chart3_end_str = datetime.date(self.end_date_datetime.year, self.end_date_datetime.month,
                                              self.end_date_datetime.day)


                            # Histograms
                            period = settings['histogram_period']
                            # First column of the 5 charts
                            title_list = [dbc.Col(width=1)]
                            pct_list = [dbc.Col(html.Div('pct_change', style={'font-size': '12px', 'margin-top': '55px',
                                                                              'margin-left': '45px',
                                                                              'transform': 'rotate(-90deg)'}), width=1)]
                            rise_list = [dbc.Col(html.Div('max_rise', style={'font-size': '12px', 'margin-top': '55px',
                                                                              'margin-left': '45px',
                                                                              'transform': 'rotate(-90deg)'}), width=1)]
                            fall_list = [dbc.Col(html.Div('max_fall', style={'font-size': '12px', 'margin-top': '55px',
                                                                              'margin-left': '45px',
                                                                              'transform': 'rotate(-90deg)'}), width=1)]


                            for p in period:
                                title_list.append(dbc.Col(f'{p} Days',style={'font-size': '12px', 'text-align': 'center'},width=2))
                                fig_pct, fig_rise, fig_fall = components.generate_histogram(df, p, 'backtest')
                                pct_list.append( dbc.Col(dcc.Graph(figure=fig_pct,config={'displayModeBar': False})
                                                         ,style={'padding':'0'},width=2))
                                rise_list.append(dbc.Col(dcc.Graph(figure=fig_rise,config={'displayModeBar': False})
                                                         ,style={'padding': '0'}, width=2))
                                fall_list.append(dbc.Col(dcc.Graph(figure=fig_fall,config={'displayModeBar': False})
                                                         ,style={'padding': '0'}, width=2))


                            title_list = html.Div(dbc.Row(title_list))
                            pct_list = html.Div(dbc.Row(pct_list))
                            rise_list = html.Div(dbc.Row(rise_list))
                            fall_list = html.Div(dbc.Row(fall_list))

                            hist_area = [html.Div(style={'height': '5px', }),
                                         title_list, pct_list, rise_list, fall_list,
                                         html.Div(style={'height': '5px', })]

                            hist_style = {'padding': '5px', 'border-radius': '5px', 'background-color': chart_bg}

                            return fig_line, filter_div, filter_dropdown_div, filter_input_div, _add_button_style,\
                                   None, chart1_button_text, chart2_button_text, hist_area, hist_style, \
                                   main_chart_area_style, chart3_dashboard_style, chart3_area_style, \
                                   chart3_start_str, chart3_end_str, chart3_chart


                        else:
                            pass    # No Line Selected


            ## Initialize After Refresh
            if not self.init:
                if not add_button_clicks:
                    if not chart1_button_click:
                        self.sort_method_current = sort_method
                        self.filter_list = []
                        self.add_button_count = 0
                        self.chart2_button_clicks = 0
                        self.chart3_button_clicks = 0
                        self.chart3_back_button_clicks = 0
                        self.save_clicks = 0
                        self.chart_type = 1
                        self.line_selected = -1
                        hist_area = []
                        hist_style = {'display': 'none'}
                        self.init = True
                        filter_dropdown_div = filter_dropdown
                        filter_input_div = filter_input


            ## Add Button Pressed
            if add_button_clicks:
                if add_button_clicks > self.add_button_count: ## Add Button Pressed
                    self.add_button_count = add_button_clicks
                    if filter_name:
                        if filter_input_value: ## Paremeter
                            if filter_name == 'exclude':
                                self.filter_list.append(['exclude', ' ', filter_input_value])
                            else:
                                self.filter_list.append([filter_name[0:-1], filter_name[-1], filter_input_value])
                            filter_div = components.update_filter_div(self.filter_list)
                            if len(self.filter_list) > 11:
                                filter_dropdown_div = filter_dropdown_disabled
                                filter_input_div = filter_input_disabled
                                _add_button_style = add_button_style_disabled
                            else:
                                filter_dropdown_div = filter_dropdown
                                filter_input_div = filter_input
                            self.init = False

            else:
                self.add_button_count = 0  # Necessary for initialization


            # Filter delete button pressed, remove one filter
            for i in range(len(vals)):
                if not vals[i] == self.button_list[i]:
                    self.filter_list.pop(i)
                    filter_div = components.update_filter_div(self.filter_list)
                    filter_dropdown_div = filter_dropdown
                    filter_input_div = filter_input
                    self.add_button_count =- 1  # Necessary for add button count
                    if len(self.filter_list) < 12:
                        filter_dropdown_div = filter_dropdown
                        filter_input_div = filter_input
                        _add_button_style = add_button_style
                    break


            # Sort Method Selected, generate Chart 1
            if sort_method:
                for para in para_checklist:
                    if para == []:
                        fig_line = components.empty_line_chart()
                        # Disable Chart 2
                        chart2_button_text = 'Strategy Analysis >'
                        self.chart_type = 1
                        hist_style = {'display': 'none'}

                        return fig_line, filter_div, filter_dropdown_div, filter_input_div, _add_button_style, \
                            None, chart1_button_text, chart2_button_text, hist_area, hist_style, \
                            main_chart_area_style, chart3_dashboard_style, chart3_area_style, \
                            chart3_start_str, chart3_end_str, chart3_chart


                current_df = result_df.copy()

                # Filter according to the filer list
                if len(self.filter_list) > 0:
                    for element in self.filter_list:
                        # print(element)
                        if element[0] == 'exclude':
                            try: code = int(element[2])
                            except: code = element[2]
                            current_df = current_df.loc[current_df['code'] != code]
                        elif element[1] == '<':
                            current_df = current_df.loc[current_df[element[0]] < float(element[2])]
                        else:
                            current_df = current_df.loc[current_df[element[0]] > float(element[2])]



                current_df = current_df.reset_index(drop=True)

                para_key_list = list(para_dict)

                if len(para_checklist) > 0:

                    and_list = []
                    for i in range(len(para_key_list)):
                        key = para_key_list[i]
                        or_list = []

                        for element in para_checklist[i]: # -1 because code / para_dict index

                            # Change back hk stock code to int
                            if key == 'code':
                                try:
                                    element = int(element)
                                except:
                                    pass

                            if element == 'True':element = True
                            elif element == 'False':element = False
                            # if key == 'tolerance':

                            _list = current_df[key] == element
                            # print(_list)
                            or_list.append(_list)
                        or_list = np.logical_or.reduce(or_list)
                        # print(key, or_list)
                        and_list.append(or_list)
                    and_list = np.logical_and.reduce(and_list)

                    df_checked = current_df.loc[pd.DataFrame(and_list, columns=['check'])['check']].reset_index(
                        drop=True).copy()
                else:

                    df_checked = current_df.copy()

                # Sort
                graph_df = components.sort_method_df(sort_method, df_checked, number_of_curves)
                self.graph_df = graph_df

                fig_line = components.generate_chart_1(graph_df, all_para_combination, intraday, summary_mode)

                # Disable Chart 2
                chart2_button_text = 'Strategy Analysis >'
                self.chart_type = 1
                hist_style = {'display': 'none'}


            return fig_line, filter_div, filter_dropdown_div,filter_input_div,_add_button_style, \
                   None, chart1_button_text, chart2_button_text, hist_area, hist_style, \
                   main_chart_area_style, chart3_dashboard_style, chart3_area_style, \
                   chart3_start_str, chart3_end_str, chart3_chart



        # Click on Chart 1 line
        @app.callback(
            Output(component_id='performance_matrix', component_property='children'),
            Output('chart2_button', 'style'),
            Input('line_chart', 'clickData'),
            State(component_id='performance_matrix', component_property='children'),
        )
        def update_matrix(clickData,performance_matrix):
            button_style = {'text-align': 'right','color':'grey','font-size':'14px'}
            self.init = False
            if clickData:
                if self.chart_type == 1:
                    i = clickData['points'][0]['curveNumber'] - 1

                    self.line_selected = i
                    button_style = {'text-align': 'right','cursor': 'pointer','font-size':'14px'}

                    reference_index = self.graph_df.iloc[i].reference_index
                    df_all_para_combination = pd.DataFrame(all_para_combination)
                    para_combination = \
                    df_all_para_combination.loc[df_all_para_combination['reference_index'] == reference_index].to_dict(
                        'records')[0]

                    save_path = plotguy.generate_filepath(para_combination=para_combination)
                    ref_code = self.graph_df.iloc[i].reference_code
                    risk_free_rate = self.graph_df.iloc[i].risk_free_rate

                    performance_matrix = components.update_performance_matrix(start_date, end_date,
                                                                              self.graph_df.iloc[i].copy(), para_dict,
                                                                              risk_free_rate, ref_code)

                    self.save_clicks = 0 # Needed for save button click tracking

                    df = self.graph_df.iloc[i].copy()
                    self.save_path = save_path
                    self.para_combination = para_combination
                    self.net_profit_to_mdd = df['net_profit_to_mdd']
                    self.net_profit = df['net_profit']
                    self.mdd_dollar = df['mdd_dollar']
                    self.mdd_pct = df['mdd_pct']
                    self.num_of_trade = df['num_of_trade']
                    try:
                        self.num_of_win = round(int(df['num_of_trade'])*float(df['win_rate'])/100)
                    except:
                        self.num_of_win = '--'
                    self.return_on_capital = df['return_on_capital']
                    self.sharpe = df['annualized_sr']
                    self.total_commission = df['total_commission']
                    self.total_commission = df['total_commission']
                    year_count = []
                    year_win_count = []
                    for year in self.year_list:
                        if not df[f'{year}_win_rate'] == '--':
                            win_count = round(int(df[str(year)])*float(df[f'{year}_win_rate'])/100)
                        else:
                            win_count = 0
                        year_count.append(df[str(year)])
                        year_win_count.append(win_count)
                    self.year_count = year_count
                    self.year_win_count = year_win_count

            return performance_matrix, button_style


        # Select All Button
        @app.callback(
            Output({'type': 'para-checklist', 'index': ALL}, 'value'),
            [Input("all-or-none", "value")],
            State({'type': 'para-checklist', 'index': ALL}, 'value'),
        )
        def select_all_none(selected_all, options):

            if selected_all == None:
                pass
            elif selected_all == ['All']:
                if self.select_all_status == False:
                    options[0] = code_list
                    self.select_all_status = True
            else:
                if self.select_all_status == True:
                    options[0] = []
                    self.select_all_status = False

            return options


        # Save Button
        @app.callback(
            Output('save_status', "children"),
            Output('save_string', "children"),
            Output('save_button', "style"),
            Input('save_button', "n_clicks"),
        )
        def save_button(save_clicks):
            if not os.path.isfile('saved_strategies.csv'):
                columns = ['path','initial','net_profit_to_mdd','net_profit',
                                      'mdd_dollar','mdd_pct','num_of_trade','num_of_win',
                                      'return_on_capital','total_commission','sharpe'
                                      ]
                for year in self.year_list:
                    columns.append(year)
                    columns.append(f'{year}_win_count')
                columns.append('para_combination')
                pd.DataFrame(columns=columns).to_csv('saved_strategies.csv', index=False)
            else:
                pass

            button_available = None
            save_status = ''
            button_string = ''
            button_style = {}

            save_path = self.save_path
            if not self.init:
                if not save_clicks: #

                    save_status = ''
                    button_string = '--'
                    button_style = {'font-size': '15px', 'margin-left': '20px', 'margin-right': '30px',
                                    'text-align': 'center', 'border-radius': '5px',
                                    'background-color': 'Grey', 'color': 'black',
                                    }

                    if self.line_selected > -1:
                        saved_df = pd.read_csv('saved_strategies.csv')
                        if not save_path in list(saved_df['path']):
                            button_available = True
                        else:
                            button_available = False


                else:
                    if save_clicks > self.save_clicks:
                        self.save_clicks = save_clicks

                        if self.line_selected > -1:
                            saved_df = pd.read_csv('saved_strategies.csv')

                            if not save_path in list(saved_df['path']):

                                para_pop = self.para_combination.copy()
                                try:
                                    para_pop.pop('df')
                                except:
                                    pass

                                try:
                                    para_pop.pop('holiday_list')
                                except:
                                    pass

                                df_dict = {'path': save_path,
                                           'para_combination':str(para_pop),
                                           'initial': self.initial_capital,
                                           'net_profit_to_mdd': self.net_profit_to_mdd,
                                           'net_profit': self.net_profit,
                                           'mdd_dollar':self.mdd_dollar,
                                           'mdd_pct':self.mdd_pct,
                                           'num_of_trade':self.num_of_trade,
                                           'num_of_win': self.num_of_win,
                                           'return_on_capital': self.return_on_capital,
                                           'sharpe': self.sharpe,
                                           'total_commission': self.total_commission,
                                           }
                                for i, year in enumerate(self.year_list):
                                    df_dict[str(year)] = self.year_count[i]
                                    df_dict[f'{year}_win_count'] = self.year_win_count[i]

                                saved_df = pd.concat([saved_df, pd.DataFrame(df_dict,index=[10])],
                                                     ignore_index=True)
                                button_available = False
                            else:
                                index = saved_df.index[saved_df['path'] == save_path].tolist()
                                saved_df = saved_df.drop(index=index)
                                button_available = True

                            saved_df.to_csv('saved_strategies.csv', index=False)

            if button_available is not None:
                if button_available:
                    save_status = ''
                    button_string = 'Save Curve'
                    button_style = {'font-size': '15px', 'margin-left': '20px', 'margin-right': '30px',
                                    'text-align': 'center', 'border-radius': '5px', 'cursor': 'pointer',
                                    'background-color': 'Yellow', 'color': 'black',
                                    }
                else:
                    save_status = '( Saved )'
                    button_string = 'Unsave Curve'
                    button_style = {'font-size': '15px', 'margin-left': '20px', 'margin-right': '30px',
                                    'text-align': 'center', 'border-radius': '5px', 'cursor': 'pointer',
                                    'background-color': 'Magenta', 'color': 'white',
                                    }



            return save_status, button_string, button_style



        return app