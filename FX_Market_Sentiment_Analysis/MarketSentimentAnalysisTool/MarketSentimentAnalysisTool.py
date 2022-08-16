# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta
from dash.dependencies import Input, Output
import asset
from asset import PeriodStart, MovingAverage, Bbands, GetSignalColor, GetIndicator, CreateDatabase, texts, CandleColor, colors


#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__)#, external_stylesheets=external_stylesheets)

server = app.server

periods = ('1M', '3M', '6M', '1Y', 'YTD', 'ALL')

codes = pd.read_csv('codes.txt', sep=',')

start_date = date(2013,1,1)
end_date = date.today()

data = CreateDatabase(codes, start_date)

app.layout = html.Div(id='layout-container', children=[

    html.Div(id='banner', children=[
        html.Img(src='/assets/logo.png'),
        html.H2('Forex Market Sentiment Analysis Tool')
        ]), 
        
    html.Div(className='row', children=[html.Div(  
        className='narrow col',
        children=[html.Div(className='panel',children=[

            html.Div(className='row', children=[
                html.Div(id='dropdown-currency', children=dcc.Dropdown(className='dropdown',
                    id='code-choice',
                    options=[{'label': i, 'value': i} for i in codes.index],
                    value=codes.index[0]
                )),
            ]),

            html.Div(className='row', children=[
                html.Div(id='dropdown-period', children=[
                    dcc.Dropdown(
                        className='dropdown',
                        id='fixed-choice',
                        options=[{'label': i, 'value': i} for i in periods],
                        value=periods[3],
                    )
                ]),
                html.Div(id='dropdown-datepicker', children=[
                    dcc.DatePickerRange(
                        className='dropdown',
                        id='custom-choice',
                        start_date=end_date - relativedelta(years=1),
                        end_date=end_date,
                        min_date_allowed=start_date,
                        max_date_allowed=end_date,
                        )
                ]),
                html.Div(id='radio-picker', children=[
                    dcc.RadioItems(
                        id='period-type',
                        options=[
                            {'label': 'Fixed', 'value': 'fixed'},
                            {'label': 'Custom', 'value':'custom'}
                        ], 
                        value='fixed'
                    )
                ])
            ]),

            html.Div(
                className='row', 
                children=dcc.Tabs(
                    children=[
                        dcc.Tab(className='tab', selected_className='selected-tab',
                            label='About',
                            children=html.Div(className='text-container', children=[
                                html.H4('Forex Market Sentiment Analysis'),
                                dcc.Markdown(children=texts['text1'])
                            ])
                        ),
                        dcc.Tab(className='tab', selected_className='selected-tab',
                            label='Legend',
                            children=html.Div(className='text-container', children=[
                                html.H4('Legend'),
                                dcc.Markdown(children=texts['text2_1']),
                                html.Br(),
                                dcc.Markdown(children=texts['text2_2']),
                                html.Br(),
                                dcc.Markdown(children=texts['text2_3'])
                            ])
                        ),
                        dcc.Tab(className='tab', selected_className='selected-tab',
                            label='Usage',
                            children=html.Div(className='text-container', children=[
                                html.H4('Market Sentiment Indicators'),
                                dcc.Markdown(children=texts['text3_1']),
                                html.Div(className='panel', children=[
                                    html.Img(id='info-table', src='/assets/table.png')
                                ]),
                                dcc.Markdown(children=texts['text3_2']),
                                html.Br(),
                                dcc.Markdown(children=texts['text3_3']),
                                html.Br(),
                                dcc.Markdown(children=texts['text3_4']),
                            ])
                        ),
                        dcc.Tab(className='tab', selected_className='selected-tab',
                            label='Sources',
                            children=html.Div(className='text-container', children=[
                                html.H4('Sources'),
                                dcc.Markdown(children=texts['text4']),
                            ])
                        )
                    ]
                )
            ),
        ])]
    ),

    html.Div(
        className='wide col',# div-right-panel',
        children=[html.Div(className='panel', children=[

            html.Div(
                dcc.Tabs(
                    children=[
                        dcc.Tab(className='tab', selected_className='selected-tab',
                            label='Historical Data', #OR HISTORICAL
                            children=html.Div(className='graph-container', children=[
                                dcc.Graph(id='graph-1'),
                                dcc.Graph(id='graph-2')
                            ])
                        ),
                        dcc.Tab(className='tab', selected_className='selected-tab',
                            label='Current Overview', #OVERVIEW?
                            children=html.Div(className='graph-container', children=[
                                html.Div(id='panel-title',
                                    children=[
                                        html.H6(id='title')
                                    ]
                                ),
                                html.Div(className='row',
                                    children=[
                                        html.Div(className='half col', children=[
                                            dcc.Graph(id='graph-3')
                                        ]),
                                        html.Div(className='half col', children=[
                                            dcc.Graph(id='graph-4')
                                        ]),
                                    ]
                                ),
                                html.Div(className='row',
                                    children=[
                                        html.Div(className='half col', children=[
                                            dcc.Graph(id='graph-5')
                                        ]),
                                        html.Div(className='half col', children=[
                                            dcc.Graph(id='graph-6')
                                        ]),
                                    ]
                                )      
                            ])
                        )
                    ]
                )
            )  
        ])
    ])])
])

@app.callback([
    #Output('fixed-choice', 'style'),
    #Output('custom-choice', 'style')
    Output('fixed-choice', 'disabled'),
    Output('custom-choice', 'disabled')
],
    [
    Input('period-type', 'value')
])
def PeriodType(selected_type):
    if(selected_type == 'fixed'):
        #return {'display': 'block'}, {'display': 'none'}
        return False, True

    else:
        #return {'display': 'none'}, {'display': 'block'}
        return True, False

@app.callback([
    Output('title', 'children'),
    Output('graph-1', 'figure'), 
    Output('graph-2', 'figure'), 
    Output('graph-3', 'figure'),
    Output('graph-4', 'figure'), 
    Output('graph-5', 'figure'),
    Output('graph-6', 'figure')
], 
    [
    Input('code-choice', 'value'),
    Input('period-type', 'value'),
    Input('fixed-choice', 'value'),
    Input('custom-choice', 'start_date'),
    Input('custom-choice', 'end_date'),
])
def UpdateGraphs(selected_code, selected_type, f_period, c_sd, c_ed):
    
    if(selected_type == 'fixed'):
        sd = PeriodStart(f_period, start_date, end_date)
        ed = end_date

    else:
        sd = c_sd
        ed = c_ed

    ohlc_all=data['OHLC'][selected_code] # FAIRE [:ed] afin de l'enlever ailleurs
    ohlc=ohlc_all.loc[sd:ed]

    cot_all=data['COT'][selected_code]
    cot=cot_all.loc[sd:ed]

    cot_tff = data['COT_TFF'][selected_code].loc[sd:ed]
    # Get changes for data of the last date
    cot_rdiff = (cot.iloc[-1] - cot.iloc[-2]) / cot.iloc[-2]
    cot_rdiff_current = (cot_all[:ed].iloc[-1] - cot_all[:ed].iloc[-2]) / cot_all[:ed].iloc[-2]
    cot_rdiff_1m = (cot_all[:ed].iloc[-1] - cot_all[:ed].iloc[-5]) / cot_all[:ed].iloc[-5]
    cot_rdiff_3m = (cot_all[:ed].iloc[-1] - cot_all[:ed].iloc[-13]) / cot_all[:ed].iloc[-13]

    #cot_rdiff_current = (data['COT'][selected_code].loc[:ed].iloc[-1] - data['COT'][selected_code].loc[:(ed-relativedelta(weeks=1))].iloc[-1]) / data['COT'][selected_code].iloc[-2]

    net_non_commercial = cot['Noncommercial Long'] - cot['Noncommercial Short']
    net_commercial = cot['Commercial Long'] - cot['Commercial Short']
    net_non_reportable = cot['Nonreportable Positions Long'] - cot['Nonreportable Positions Short']

    difference = net_non_commercial - net_commercial

    net_dealer_intermediary = cot_tff['Dealer Longs'] - cot_tff['Dealer Shorts']
    net_asset_manager = cot_tff['Asset Manager Longs'] - cot_tff['Asset Manager Shorts']
    net_leveraged_funds = cot_tff['Leveraged Funds Longs'] - cot_tff['Leveraged Funds Shorts']
    net_other_reportables = cot_tff['Other Reportable Longs'] - cot_tff['Other Reportable Shorts']

    percentage_trades_long = (cot_all['Noncommercial Long'][:ed]/(cot_all['Noncommercial Long'][:ed] + cot_all['Noncommercial Short'][:ed])) * 100
    percentage_trades_short = 100 - percentage_trades_long

    #percentage_trades_long = (data['COT'][selected_code]['Noncommercial Long']/(data['COT'][selected_code]['Noncommercial Long'] + data['COT'][selected_code]['Noncommercial Short'])) * 100
    #percentage_trades_short = 100 - percentage_trades_long

    


    data_ohlc = dict(
        type='candlestick',
        open=ohlc['Open'],
        high=ohlc['High'],
        low=ohlc['Low'],
        close=ohlc['Settle'],
        x=ohlc.index,
        yaxis='y3',
        name=f'{selected_code} Futures',
        increasing=dict(line=dict(color=colors['increasing'])),
        decreasing=dict(line=dict(color=colors['decreasing']))
    )


    title=f'Non-commercial overview for period {sd} - {ed}'

    #================================================================================FIG 1=====================================================================================

    data1 = [data_ohlc.copy()]

    # The Commitment of Traders Report

    # Commercials, consisting of Producer/Merchant/Processor/User and Swap Dealers
    data1.append(
        dict(
            x=cot.index,
            y=net_commercial,
            marker=dict(color=colors['red1']),
            line=dict(shape='vh'),
            yaxis='y2',
            name='Commercial Traders'
        )
    )

    # Non-Commercials or Large Speculators, consisting of Managed Money and
    # Other Reportables
    data1.append(
        dict(
            x=cot.index,
            y=net_non_commercial,
            marker=dict(color=colors['blue1']),
            line=dict(shape='vh'),
            yaxis='y2',
            name='Non-commercial Traders'
        )
    )

# Small Speculators
    data1.append(
        dict(
            x=cot.index,
            y=net_non_reportable,
            marker=dict(color=colors['green1']),
            line=dict(shape='vh'),
            yaxis='y2',
            name='Non-reportable Positions'
        )
    )

# The Commitments of Traders Financial Traders (TFF) Report

    data1.append(
        dict(
            x=cot_tff.index,
            y=net_dealer_intermediary,
            marker=dict(color=colors['blue2']),
            line=dict(shape='vh'),
            yaxis='y',
            name='Dealer/Intermediary'
        )
    )

    data1.append(
        dict(
            x=cot_tff.index,
            y=net_asset_manager,
            marker=dict(color=colors['red2']),
            line=dict(shape='vh'),
            yaxis='y',
            name='Asset Manager'
        )
    )

    data1.append(
        dict(
            x=cot_tff.index,
            y=net_leveraged_funds,
            marker=dict(color=colors['green2']),
            line=dict(shape='vh'),
            yaxis='y',
            name='Leveraged Funds'
        )
    )

    data1.append(
        dict(
            x=cot_tff.index,
            y=net_other_reportables,
            marker=dict(color=colors['orange2']),
            line=dict(shape='vh'),
            yaxis='y',
            name='Other Reportables'
        )
    )


    layout1 = dict(
        title=f"Continuous {selected_code.split('-')[1]} Futures",
        height=600,
        xaxis=dict(
            rangeslider=dict(visible=False),
            showgrid=False, #True
        ), 
        yaxis=dict(
            domain=[0, 0.1],
            title='TFF',
            showgrid=True,
            #zeroline=False,
            ), 
        yaxis2=dict(
            domain=[0.15, 0.25],
            title='COT',
            showgrid=True,
            #zeroline=False,
            ),
        yaxis3=dict(
            domain=[0.3, 1], 
            title='Daily',
            showgrid=True,
            ),
        legend=dict(
            orientation='h', 
            x=0, 
            y=-0.2, 
            yanchor='bottom', 
            traceorder='normal'),
        annotations=[
            dict(
                x=0,
                y=1.025,
                xref='paper',
                yref='paper',
                text=f"Open: {ohlc['Open'][-1]}  High: {ohlc['High'][-1]}  Low: {ohlc['Low'][-1]}  Close: {ohlc['Settle'][-1]}",
                showarrow=False,
                font=dict(color=CandleColor(ohlc.iloc[-1]), size=10)
            ), 

            dict(
                x=0, 
                y=0.25,
                xanchor='left',
                xref='paper',
                yref='paper',
                text='<b>Net Positions (Longs - Shorts)</b>',
                font=dict(size=10),
                showarrow=False
            ),

            dict(
                x=1, 
                y=0.25,
                xanchor='right',
                xref='paper',
                yref='paper',
                text=f'Commercial: {int(net_commercial[-1])}  Non-commercial: {int(net_non_commercial[-1])}  Non-reportable: {int(net_non_reportable[-1])}',
                font=dict(size=10),
                showarrow=False
            ),

            dict(
                x=1,
                y=0.1,
                xanchor='right',
                xref='paper',
                yref='paper',
                text=f'Dealer/Intermediary: {int(net_dealer_intermediary[-1])}  Asset Manager: {int(net_asset_manager[-1])}  Leveraged Funds: {int(net_leveraged_funds[-1])}  Other Reportables: {int(net_other_reportables[-1])}',
                font=dict(size=10),
                showarrow=False,
            )
        ]
    )


    fig1 = dict(data=data1, layout=layout1)

    #================================================================================FIG 2=====================================================================================

    data2 = [data_ohlc.copy()]

    data2.append(
        dict(
            x=cot.index,
            y=percentage_trades_long[sd:ed],
            type='scatter',
            mode='lines',
            line=dict(width=0.5, color=colors['long']),
            stackgroup='one',
            yaxis='y2',
            name='% Long',
            hovertemplate='%{x}, %{y}%',
        )

    )

    data2.append(
        dict(
            x=cot.index,
            #y=percentage_trades_short,
            y=percentage_trades_short[sd:ed],
            type='scatter',
            mode='lines',
            line=dict(width=0.5, color=colors['short']),
            stackgroup='one',
            yaxis='y2',
            name='% Short',
            hovertemplate='%{x}, %{y}%',
        )

    )

    data2.append(
        dict(
            x=cot.index,
            y=net_commercial,
            marker=dict(color=colors['red1']),
            line=dict(shape='vh'),
            yaxis='y',
            name='Commercial Traders'
        )
    )

    data2.append(
        dict(
            x=cot.index,
            y=net_non_commercial,
            marker=dict(color=colors['blue1']),
            line=dict(shape='vh'),
            yaxis='y',
            name='Non-commercial Traders'
        )
    )

     
    layout2 = dict(
        title=f"Continuous {selected_code.split('-')[1]} Non-commercial Positioning", 
        height=600,
        #hovermode='x unified',
        xaxis=dict(
            showgrid=False
        ),
        yaxis=dict(
            domain=[0, 0.15],
            title='COT',
            #side='right'
        ), 
        yaxis2=dict(
            domain=[0.2, 1],
            overlaying='y3', 
            side='right',
            hoverformat='.1f',
            #title='(%)', 
            zeroline=False), 
        yaxis3=dict(
            domain=[0.2, 1], 
            title='Daily', 
            zeroline=False), 
        legend=dict(
            orientation='h', 
            x=0, 
            y=-0.5, 
            yanchor='bottom', 
            traceorder='normal'
        ),
        annotations=[
            dict(
                x=0,
                y=1.025,
                xref='paper',
                yref='paper',
                text=f"Open: {ohlc['Open'][-1]}  High: {ohlc['High'][-1]}  Low: {ohlc['Low'][-1]}  Close: {ohlc['Settle'][-1]}",
                showarrow=False,
                font=dict(color=CandleColor(ohlc.iloc[-1]), size=10),
            ), 

            dict(
                x=1,
                y=1.025,
                xref='paper',
                yref='paper',
                xanchor='right',
                text=f'Long: {round(percentage_trades_long[sd:ed][-1],1)}%  Short: {round(percentage_trades_short[sd:ed][-1],1)}%',
                font=dict(size=10),
                showarrow=False,
            ),

            dict(
                x=0,
                y=0.15,
                xref='paper',
                yref='paper',
                xanchor='left',
                text='<b>Net Positions (Longs - Shorts)</b>',
                font=dict(size=10),
                showarrow=False,
            ),

            dict(
                x=1,
                y=0.15,
                xref='paper',
                yref='paper',
                xanchor='right',
                text=f'Commercial: {int(net_commercial[-1])}  Non-commercial: {int(net_non_commercial[-1])}',
                font=dict(size=10),
                showarrow=False,
            )
        ]
    )
        

    fig2 = dict(data=data2, layout=layout2)

    
    #========================================================FIG 3=====================================================================================

    data3 = [

        dict(
            type='bar',
            orientation='h',
            x=[percentage_trades_long[-13], percentage_trades_long[-5], percentage_trades_long[-2], percentage_trades_long[-1]], #non commercial
            y=['1 Quarter', '1 Month', '1 Week', 'Current'],
            name='Current percentage client position long/short', 
            hoverinfo='none',
            marker=dict(
                color=colors['long']
            )
        ),

        dict(
            type='bar',
            orientation='h',
            x=[percentage_trades_short[-13], percentage_trades_short[-5], percentage_trades_short[-2], percentage_trades_short[-1]], #non commercial
            y=['1 Quarter', '1 Month', '1 Week', 'Current'],
            name='Current percentage client position long/short', 
            hoverinfo='none',
            marker=dict(
                color=colors['short']
            )
        ),
    ]

    annotations3 = [

    dict(
        xref='paper', yref='paper',
        x=0.2, y=1.3,
        xanchor='center',
        text='<b>LONG</b>',
        showarrow=False
    ),

    dict(
        xref='paper', yref='paper',
        x=1, y=1.3,
        xanchor='center',
        text='<b>SHORT</b>',
        showarrow=False
    ),

    dict(
        xref='paper', yref='y',
        x=-0.1, y='Current',
        xanchor='left',
        text='<b>Current</b>', 
        showarrow=False
    ),

    dict(
        xref='paper', yref='y',
        x=0.2, y='Current',
        xanchor='center',
        text=f'<b>{round(percentage_trades_long[-1],1)}%</b>',
        font=dict(color=colors['long']),
        showarrow=False
    ),

    dict(
        xref='paper', yref='y',
        x=1, y='Current',
        xanchor='center',
        text=f'<b>{round(percentage_trades_short[-1],1)}%</b>',
        font=dict(color=colors['short']),
        showarrow=False
    ),

    dict(
        xref='paper', yref='y',
        x=-0.1, y='1 Week',
        xanchor='left',
        text='<b>1 Week</b>',
        showarrow=False
    ),

    dict(
        xref='paper', yref='y',
        x=0.2, y='1 Week',
        xanchor='center',
        text=f'<b>{round(percentage_trades_long[-2],1)}%</b>',
        font=dict(color=colors['long']),
        showarrow=False
    ),

    dict(
        xref='paper', yref='y',
        x=1, y='1 Week',
        xanchor='center',
        text=f'<b>{round(percentage_trades_short[-2],1)}%</b>',
        font=dict(color=colors['short']),
        showarrow=False
    ),

    dict(
        xref='paper', yref='y',
        x=-0.1, y='1 Month',
        xanchor='left',
        text='<b>1 Month</b>',
        showarrow=False
    ),

    dict(
        xref='paper', yref='y',
        x=0.2, y='1 Month',
        xanchor='center',
        text=f'<b>{round(percentage_trades_long[-5],1)}%</b>',
        font=dict(color=colors['long']),
        showarrow=False
    ),

    dict(
        xref='paper', yref='y',
        x=1, y='1 Month',
        xanchor='center',
        text=f'<b>{round(percentage_trades_short[-5],1)}%</b>',
        font=dict(color=colors['short']),
        showarrow=False
    ),

    dict(
        xref='paper', yref='y',
        x=-0.1, y='1 Quarter',
        xanchor='left',
        text='<b>1 Quarter</b>',
        showarrow=False
    ),

    dict(
        xref='paper', yref='y',
        x=0.2, y='1 Quarter',
        xanchor='center',
        text=f'<b>{round(percentage_trades_long[-13],1)}%</b>',
        font=dict(color=colors['long']),
        showarrow=False
    ),

    dict(
        xref='paper', yref='y',
        x=1, y='1 Quarter',
        xanchor='center',
        text=f'<b>{round(percentage_trades_short[-13],1)}%</b>',
        font=dict(color=colors['short']),
        showarrow=False
    ),

    ]
       

    layout3 = dict(
        title=f'Client Positioning (as of {cot.index[-1].date()})',
        height=250,
        xaxis=dict(
            domain=[0.3, 0.9],
            range=(0,100),
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False, 
            fixedrange=True
            ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
            fixedrange=True
        ),
        showlegend=False,
        barmode='stack',
        annotations=annotations3)

    fig3 = dict(data=data3, layout=layout3)

    #========================================================FIG 3bis=====================================================================================

    data4 = [

        dict(
            type='bar',
            orientation='h',
            x=[0,0,0],
            y=['1 Quarter', '1 Month', '1 Week'],
            hoverinfo='none',
        )
    ]

    #ohlc_prior_1w = ohlc_all.loc[:ed].index[-1] - relativedelta(weeks=1)
    #ohlc_prior_1m = ohlc_all.loc[:ed].index[-1] - relativedelta(months=1)
    #ohlc_prior_3m = ohlc_all.loc[:ed].index[-1] - relativedelta(months=3)

    ohlc_prior_1w = cot_all.loc[:ed].index[-2]
    ohlc_prior_1m = cot_all.loc[:ed].index[-5]
    ohlc_prior_3m = cot_all.loc[:ed].index[-13]

    #text_weekly, color_weekly = GetSignalColor(ohlc_all['Settle'][:ed][-1], ohlc_all['Settle'][:ohlc_prior_1w][-1])
    text_weekly, color_weekly = GetSignalColor(ohlc['Settle'][cot.index[-1]], ohlc_all['Settle'][:ohlc_prior_1w][-1])
    text_monthly, color_monthly = GetSignalColor(ohlc['Settle'][cot.index[-1]], ohlc_all['Settle'][:ohlc_prior_1m][-1])
    text_quarterly, color_quarterly = GetSignalColor(ohlc['Settle'][cot.index[-1]], ohlc_all['Settle'][:ohlc_prior_3m][-1])

    annotations4 = [

        dict(
            xref='paper', yref='paper',
            x=0.2, y=1.3,
            xanchor='center',
            text='<b>LONGS</b>',
            showarrow=False
        ),

        dict(
            xref='paper', yref='paper',
            x=0.4, y=1.3,
            xanchor='center',
            text='<b>SHORTS</b>',
            showarrow=False
        ),

        dict(
            xref='paper', yref='paper',
            x=0.6, y=1.3,
            xanchor='center',
            text='<b>OI</b>',
            showarrow=False
        ),

        dict(
            xref='paper', yref='paper',
            x=0.9, y=1.3,
            xanchor='center',
            text='<b>FUTURES*</b>',
            showarrow=False
        ),

        dict(
            xref='paper', yref='paper',
            x=0, y=-0.3,
            xanchor='left',
            text='<i>* Evolution of futures closing prices</i>',
            font=dict(size=10),
            showarrow=False
        ),

        dict(
            xref='paper', yref='y',
            x=-0.1, y='1 Week',
            xanchor='left',
            text='<b>Weekly</b>',
            showarrow=False
        ),

        dict(
            xref='paper', yref='y',
            x=0.2, y='1 Week',
            xanchor='center',
            text=f"{round(cot_rdiff_current['Noncommercial Long'] * 100, 1)}%",
            showarrow=False
        ),

        dict(
            xref='paper', yref='y',
            x=0.4, y='1 Week',
            xanchor='center',
            text=f"{round(cot_rdiff_current['Noncommercial Short'] * 100, 1)}%",
            showarrow=False
        ),

        dict(
            xref='paper', yref='y',
            x=0.6, y='1 Week',
            xanchor='center',
            text=f"{round(cot_rdiff_current['Open Interest'] * 100, 1)}%",
            showarrow=False
        ),

        dict(
            type='scatter',
            mode='marker',
            xref='paper', yref='y',
            x=0.9, y='1 Week',
            xanchor='center',
            #text=f"{ohlc_all['Settle'][:ohlc_prior_1w][-1]} ➧ {ohlc_all['Settle'][-1]}",
            text=f"{ohlc_all['Settle'][:ohlc_prior_1w][-1]} ➧ {ohlc['Settle'][cot.index[-1]]}",
            font=dict(color=color_weekly),
            showarrow=False,
        ),

        dict(
            xref='paper', yref='y',
            x=-0.1, y='1 Month',
            xanchor='left',
            text='<b>Monthly</b>',
            showarrow=False
        ),

        dict(
            xref='paper', yref='y',
            x=0.2, y='1 Month',
            xanchor='center',
            text=f"{round(cot_rdiff_1m['Noncommercial Long'] * 100, 1)}%",
            showarrow=False
        ),

        dict(
            xref='paper', yref='y',
            x=0.4, y='1 Month',
            xanchor='center',
            text=f"{round(cot_rdiff_1m['Noncommercial Short'] * 100, 1)}%",
            showarrow=False
        ),

        dict(
            xref='paper', yref='y',
            x=0.6, y='1 Month',
            xanchor='center',
            text=f"{round(cot_rdiff_1m['Open Interest'] * 100, 1)}%",
            showarrow=False
        ),

        dict(
            xref='paper', yref='y',
            x=0.9, y='1 Month',
            xanchor='center',
            text=f"{ohlc_all['Settle'][:ohlc_prior_1m][-1]} ➧ {ohlc['Settle'][cot.index[-1]]}", 
            font=dict(color=color_monthly),
            showarrow=False
        ),

        dict(
            xref='paper', yref='y',
            x=-0.1, y='1 Quarter',
            xanchor='left',
            text='<b>Quarterly</b>',
            showarrow=False
        ),

        dict(
            xref='paper', yref='y',
            x=0.2, y='1 Quarter',
            xanchor='center',
            text=f"{round(cot_rdiff_3m['Noncommercial Long'] * 100, 1)}%",
            showarrow=False
        ),

        dict(
            xref='paper', yref='y',
            x=0.4, y='1 Quarter',
            xanchor='center',
            text=f"{round(cot_rdiff_3m['Noncommercial Short'] * 100, 1)}%",
            showarrow=False
        ),

        dict(
            xref='paper', yref='y',
            x=0.6, y='1 Quarter',
            xanchor='center',
            text=f"{round(cot_rdiff_3m['Open Interest'] * 100, 1)}%",
            showarrow=False
        ),

        dict(
            xref='paper', yref='y',
            x=0.9, y='1 Quarter',
            xanchor='center',
            text=f"{ohlc_all['Settle'][:ohlc_prior_3m][-1]} ➧ {ohlc['Settle'][cot.index[-1]]}", 
            font=dict(color=color_quarterly),
            showarrow=False
        ),
    ]

    layout4 = dict(
        title=f'Change in Position (as of {cot.index[-1].date()})',
        height=250,
        xaxis=dict(
            range=(0,1),
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False, 
            fixedrange=True,
        ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
            fixedrange=True
        ),
        showlegend=False,
        annotations=annotations4)

    fig4 = dict(data=data4, layout=layout4)

    #========================================================FIG 4=====================================================================================

    df = pd.DataFrame(difference)

    data5 = [
        dict(
            type='scatter',
            x=difference.index,
            y=difference,
            name='Difference',
            line=dict(colors=colors['increasing'])

        ),
        dict(
            type='scatter',
            x=[sd,ed],
            y=[difference[-1], difference[-1]],
            name='Actual',
            line=dict(color=colors['decreasing'], dash='dash')
        )
    ]

    layout5 = dict(
        title='Difference Net Non-Commercial and Net Commercial', 
        height=250, 
        legend=dict(orientation='h', x=0.15, y=-0.6, yanchor='bottom', traceorder='normal'),
        annotations=[
            dict(
                xref='paper',
                yref='paper',
                x=0,
                y=1.3,
                xanchor='left',
                text=f'Min: {int(min(difference))}  Max: {int(max(difference))}  Actual: {int(difference[-1])}',
                showarrow=False,
                font=dict(size=10)
            )
        ]
    )

    fig5 = dict(data=data5, layout=layout5)

    #========================================================FIG 5=====================================================================================
    indicator = GetIndicator(difference)

    indicator_colors=('green2', 'orange2', 'red2')

    data6=[
        dict(
            type='bar',
            orientation='h',
            x=[-50, -30, -20, 50, 30, 20],
            y=[0.5]*6,
            marker=dict(
                color=[colors[color] for color in indicator_colors] * 2
            ),
            showlegend=False,
            hoverinfo='none'
        ),

        dict(
            type='scatter',
            mode='marker',
            x=[indicator],
            yref='paper',
            y=[0.9],
            yaxis='y',
            hoverinfo='x',
            marker=dict(size=10, color='green', symbol='triangle-down', line=dict(width=1,color=colors['green1']))
        ),

        dict(
            type='scatter',
            mode='marker',
            x=[indicator],
            yref='paper',
            y=[0.1],
            yaxis='y',
            hoverinfo = 'none',
            marker=dict(size=10, color='green', symbol='triangle-up', line=dict(width=1, color=colors['green1']))
        )
    ]


    layout6 = dict(
        title='Reversal Indicator', 
        height=250,
        xaxis=dict(
            domain=[0.1, 0.9],
            range=(-110,110),
            showgrid=False, 
            zeroline=False, 
            fixedrange=True, 
            tickmode='array', 
            tickvals=[-100,-80,-50,0,50,80,100]), 
        yaxis=dict(
            domain=[0.3,0.7],
            range=(0,1),
            showgrid=False, 
            showline=False, 
            showticklabels=False, 
            zeroline=False, 
            fixedrange=True), 
        showlegend=False,
        barmode='relative', 
        annotations=[
            dict(xref='paper', yref='paper', x=0, y=0.5, xanchor='center', text='<b>Extreme<br>Low</b>', showarrow=False),
            dict(xref='paper', yref='paper', x=1, y=0.5, xanchor='center', text='<b>Extreme<br>High</b>', showarrow=False),
        ]
    )


    fig6 = dict(data=data6, layout=layout6)



    return title, fig1, fig2, fig3, fig4, fig5, fig6


if __name__ == '__main__':
    app.run_server(debug=False)

