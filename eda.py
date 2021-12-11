import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input, State
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import pandas_datareader.data as web
import datetime
import pickle
import numpy as np
from geopy import geocoders 
from geopy.geocoders import Nominatim


df = pd.read_csv(r'NYC_output (1).csv')

copy_df = df.copy()
copy_df = copy_df.sort_values(by='name')
keys = copy_df['name'].unique()
values = copy_df['city_code'].unique()
res = {keys[i]: values[i] for i in range(len(keys))}
res['All'] = 'None'

ALLOWED_TYPES = (
    "text", "number", "password", "email", "search",
    "tel", "url", "range", "hidden",
)

conditions = [
    (df['trip_distance'] <= 5),
    (df['trip_distance'] > 5) & (df['trip_distance'] <= 10),
    (df['trip_distance'] > 10) & (df['trip_distance'] <= 15),
    (df['trip_distance'] > 15)
    ]

# create a list of the values we want to assign for each condition
values = ['[0-5mile]', '[5-10mile]', '[10-15mile]', '[15 above]']

# create a new column and use np.select to assign values to it using our lists as arguments
df['tier'] = np.select(conditions, values)

conditions = [
    (df['pickup_counts'] <= 5000),
    (df['pickup_counts'] > 5000) & (df['pickup_counts'] <= 7000),
    (df['pickup_counts'] > 7000) & (df['pickup_counts'] <= 9000),
    (df['pickup_counts'] > 9000)& (df['pickup_counts'] <= 11000),
    (df['pickup_counts'] > 11000),
    ]

# create a list of the values we want to assign for each condition
values = ['[0-5000]', '[5000-7000]', '[7000-9000]', '[9000-11000]', '[11000 above]']

# create a new column and use np.select to assign values to it using our lists as arguments
df['pickup_bin'] = np.select(conditions, values)

df1=df.loc[df['name'].isin(['Manhattan','New York City'])]
df4=df.groupby(
     ['name']
 ).agg(
     trip_distance = ('trip_distance','sum'),
 ).reset_index()
df5=df.groupby(
     ['name']
 ).agg(
     total_amount = ('total_amount','sum'),
 ).reset_index()

def generateGraph():
    fig = px.density_mapbox(df, lat='pickup_latitude', lon='pickup_longitude', z='pickup_counts', radius=10,
                        zoom=10,
                        mapbox_style="stamen-terrain", width=1000, height=600)
    return fig

def revenueGraph():
    grp_city = df.groupby('city_code')
    g = df.groupby('city_code').mean()
    normalized_df = (g - g.min())/(g.max() - g.min()) * 100
    list_b = []
    for i in grp_city.groups.keys():
        list_b.append(i)

    fig = px.line(x = list_b, y = g['total_amount'], labels={'x': 'CITY CODE', 'y': 'AVERGAE FARE'})
    fig.add_bar(x = list_b, y = normalized_df['pickup_counts'])

    return fig

# https://www.bootstrapcdn.com/bootswatch/
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )


# Layout section: Bootstrap (https://hackerthemes.com/bootstrap-cheatsheet/)
# ************************************************************************
app.layout = dbc.Container([

    dbc.Row(
        dbc.Col(html.H1("TAXI DATA ANALYTICS DASHBOARD",
                        className='text-center text-primary mb-4'),
                width=12)
    ),
    dbc.Row([
        dbc.Col(
             html.Div([
                    dbc.Label("Select the City Code"),
                    dcc.Dropdown(id="city_code", value="None",
                    options=[{'label':key, 'value':value}
                                  for key, value in res.items()],
                    )]
             )
        )
    ]),
    dbc.Row([
        dbc.Col([
                html.Br(),
                html.Div([
                    
                    dbc.Label("Select the attribute on X-axis"),
                        dcc.Dropdown(id="x_axis_1", value='pickup_hour', clearable=False,
                                        options=[
                                            {'label': 'Hours', 'value': 'pickup_hour'},
                                            {'label': 'Passenger Count', 'value': 'passenger_count'},
                                       ],
                                ),
                ])
        
        ]),
        dbc.Col([
                html.Br(),
                html.Div([
                    dbc.Label("Select the attribute on Y-axis"),
                    dcc.Dropdown(id="y_axis_1", value='pickup_counts', clearable=False,
                                        options=[
                                            {'label': 'Pickups', 'value': 'pickup_counts'},
                                            {'label': 'Fare Amount', 'value': 'fare_amount'},
                                            {'label': 'Total Amount', 'value': 'total_amount'},
                                            {'label': 'Total Trip Distance', 'value': 'trip_distance'},                                            
                                       ],
                                ),
                
                ]),
        ])
    ]),
    dbc.Row([
        html.Div([dbc.Row([ 
               
                dbc.Col([
                    html.Br(),
                    dcc.RadioItems(id='radio',
                                options=[{'label': 'Bar Chart', 'value': 'bar_chart'},
                                            {'label': 'Line Chart', 'value': 'line_chart'}],
                                value='line_chart',
                                labelClassName="mr-3",
                                labelStyle = {'display': 'block','margin-right': '70px','font-weight': 300}), ],
                    width={'size': 11, 'offset': 1},
                    xs=12, sm=12, md=12, lg=12, xl=12
                    ),])
                ],style={'left': '50%', 'float': 'left', 'width': '400px', 'display': 'inline-block', 'marginLeft': '40px',
                    'color': '#0D47A1'}),
                
                html.Div(style={'margin-top': '50px'}),
                html.Div(dbc.Row([
                    dcc.Graph(id='graph')
                ])),    

    ]),
    html.Br(),
    html.H3("Estimated average fare amount and number of pickups in each region", className="justify-content-center text-center"),
    html.Div(
        dcc.Graph(id='graph_static2', figure=revenueGraph()),
        # dcc.Graph(id='graph_static1', figure=generateGraph())
    ),     

    html.Div([
        html.Br(),
    ]),

    html.Br(),
    html.H3("No. of pickups per distance bin in each region", className="justify-content-center text-center"),

    dbc.Row([
        dbc.Col(
             html.Div([
                    dbc.Label("Select the City Code"),
                    dcc.Dropdown(id="city_code_new", value="NYC",
                    options=[{'label':key, 'value':value}
                                  for key, value in res.items()],
                    )]
             )
        )
    ]),

    html.Div(
            dcc.Graph(id="bingraph",figure={}),
    ),

    html.Br(),
    html.H3("Hourly change of trip amount along with pickup amounts for each region", className="justify-content-center text-center"),
    html.Div([
        html.Br(),
        dcc.Graph(id='animation',figure=px.scatter(df, x="city_code", y="total_amount", animation_frame="pickup_hour", size="pickup_counts", animation_group="name", color="city_code"), ),
    ]),

    html.Br(),
    html.H3("Revenue generated for each region", className="justify-content-center text-center"),
    html.Div([
        html.Br(),
        dcc.Graph(id='treemap', figure=px.treemap(df, path=[px.Constant("all"), 'city_code', 'tier', 'pickup_bin'], values='total_amount'))
    ]),

    html.Br(),
    html.H3("Density based hotspots in all regions", className="justify-content-center text-center"),
    html.Div(
        dcc.Graph(id='graph_static1', figure=generateGraph())
    ),


    html.Br(),
    html.H3("TAXI FARE AMOUNT PREDICTION", className="justify-content-center text-center"),
    
    dbc.Row([
        dbc.Col([
            html.Div([

                dbc.Label("No. of Passengers"),
                dcc.Dropdown(id="pass_count", value=2,
                                    options=[{'label':x, 'value':x}
                                    for x in range(1, 7)],
                                ),
            ])
        ]),
        dbc.Col([
            html.Div([

                dbc.Label("PICKUP HOUR"),
                dcc.Dropdown(id="ml_pickup_hr", value='5', clearable=False,
                                        options=[{'label':x, 'value':x}
                                  for x in range(0, 24)],
                                ),
               
            ])
        ]),
        dbc.Col([
            html.Div([
               
                dbc.Label("DROFF HOUR"),
                dcc.Dropdown(id="ml_dropoff_hr", value='5', clearable=False,
                                        options=[{'label':x, 'value':x}
                                  for x in range(0, 24)],
                                ),
            ])
        ]),
    ]),

    dbc.Row([
        html.Br()
    ]),

    dbc.Row([
        
        dbc.Col([
            html.Div([

                    dbc.Label("TRIP DISTANCE"),
                    html.Br(),
                    dcc.Input(
                    id='my_trip_dist',
                    type='text',          # changes to input are sent to Dash server only on enter or losing focus
                    name='Trip Distance',             # the name of the control, which is submitted with the form data
                ),
            ])
        ]),
        dbc.Col([
            html.Div([

                    dbc.Label("Street Name pickup address"),
                    html.Br(),
                    dcc.Input(
                    id='lat',
                    type='text',          # changes to input are sent to Dash server only on enter or losing focus
                    name='latitude',             # the name of the control, which is submitted with the form data
                ),
               
            ])
        ]),
        dbc.Col([
            html.Div([

                    dbc.Label("Zipcode of the pickup address"),
                    html.Br(),
                    dcc.Input(
                    id='lon',
                    type='text',          # changes to input are sent to Dash server only on enter or losing focus
                    name='longitude',             # the name of the control, which is submitted with the form data
                ),
               
            ])
        ]),
        dbc.Col([
            html.Div([
                dbc.Button('Predict', id='predict', color='primary', className="justify-content-center text-center"),
                ], className="d-flex justify-content-start"),
        ])
    ]),
    html.Br(),
    dbc.Row([
        html.Br(),
        html.Div([
            html.Div([
                    html.H5("PREDICTED TAXI FARE", className = "card-title mb-2 text-center"),
                    dcc.Input(id = 'result', className = "card-text mb-2 text-muted d-flex align-items-center text-center font-weight-bolder", style = {'border': 'none', 'font-size': "20px"})
            ], className = "card-body")
            
        ], className = "card d-flex justify-content-center d-flex align-items-center")

    ],  style={"border": "none", 'left': '1%', 'float': 'left', 'width': '400px', 'display': 'inline-block',
                   'marginLeft': '350px',
                   'marginRight': '10px', 'margin-top': '30px', 'margin-bottom': '30px',
                   'color': '#0D47A1'})           

])

@app.callback(
Output('graph', 'figure'),
[
  Input('city_code', 'value'),
  Input('x_axis_1', 'value'),
  Input('y_axis_1', 'value'),
  Input('radio', 'value')
])

def render_comparision_graph(city_code, x_axis_1, y_axis_1, radio):
    
    print(radio)
    if(city_code != 'None'):
        grouped_data = df.groupby('city_code')
        temp = grouped_data.get_group(city_code)
        ride_city_grp = temp.groupby('passenger_count')
        if(radio == 'line_chart'):
            print("LINE")
            if(x_axis_1 == 'pickup_hour'):
                if(y_axis_1 == 'pickup_counts' or y_axis_1 == 'fare_amount' or y_axis_1 == 'total_amount' or y_axis_1 == 'trip_distance'):
                    fig = px.line(temp, x = x_axis_1, y = y_axis_1, labels={'x': x_axis_1, 'y': y_axis_1}).update_traces(mode='lines+markers')
                else:
                    fig = px.line(temp, x='pickup_hour', y = 'pickup_counts').update_traces(mode='lines+markers')

                return fig
            else:
                if(y_axis_1 == 'pickup_counts' or y_axis_1 == 'fare_amount' or y_axis_1 == 'total_amount' or y_axis_1 == 'trip_distance'):
                    list_a = []
                    list_b = []
                    for i in ride_city_grp.groups.keys():
                        t = ride_city_grp.get_group(i)
                        list_a.append(t[y_axis_1].mean())
                        list_b.append(i)
                    fig = px.line(x = list_b, y = list_a, labels={'x': x_axis_1, 'y': y_axis_1}).update_traces(mode='lines+markers')
                    return fig
        
        elif(radio == 'bar_chart'):
            if(x_axis_1 == 'pickup_hour'):
                if(y_axis_1 == 'pickup_counts' or y_axis_1 == 'fare_amount' or y_axis_1 == 'total_amount' or y_axis_1 == 'trip_distance'):
                    print(max(temp[x_axis_1]), max(temp[y_axis_1]))
                    fig = px.bar(x = temp[x_axis_1], y = temp[y_axis_1], labels={'x': x_axis_1, 'y': y_axis_1})
                    fig.update_traces(marker_color = 'rgba(0,0,250, 0.5)',
                        marker_line_width = 0,
                        selector=dict(type="bar"))
                else:
                    # df['pickup_counts'] = df['pickup_hour'].map(df['pickup_hour'].value_counts())
                    fig = px.bar(temp, x = x_axis_1, y = y_axis_1, labels={'x': x_axis_1, 'y': y_axis_1})
                    fig.update_traces(marker_color = 'rgba(0,0,250, 0.5)',
                        marker_line_width = 0,
                        selector=dict(type="bar"))
                return fig
            else:
                if(y_axis_1 == 'pickup_counts' or y_axis_1 == 'fare_amount' or y_axis_1 == 'total_amount' or y_axis_1 == 'trip_distance'):
                    list_a = []
                    list_b = []
                    for i in ride_city_grp.groups.keys():
                        t = ride_city_grp.get_group(i)
                        list_a.append(t[y_axis_1].mean())
                        list_b.append(i)
                    fig = px.bar(temp, x = list_b, y = list_a, labels={'x': x_axis_1, 'y': y_axis_1})
                    fig.update_traces(marker_color = 'rgba(0,0,250, 0.5)',
                    marker_line_width = 0,
                    selector=dict(type="bar"))
                    return fig
    else:
        if(x_axis_1 == 'pickup_hour'):
            # print(df)
            df['pickup_counts'] = df['pickup_hour'].map(df['pickup_hour'].value_counts())
            fig = px.line(df, x='pickup_hour', y='pickup_counts', color = 'city_code').update_traces(mode='lines+markers')
            return fig
        else:
            rides = df.groupby('passenger_count')
            list_a = []
            list_b = []
            for i in rides.groups.keys():
                t = rides.get_group(i)
                list_a.append(t[y_axis_1].mean())
                list_b.append(i)
            fig = px.line(x = list_b, y = list_a, labels={'x': x_axis_1, 'y': y_axis_1})
            return fig


@app.callback(
    Output('bingraph','figure'),
    Input('city_code_new','value')
)
def bin_graph(city):
    graph_dataframe=df[df['city_code'] == city]

    graph_dataframe['pickup_counts'] = graph_dataframe['pickup_hour'].value_counts()
    bar_fig=px.bar(graph_dataframe,x='tier',y='pickup_counts')
    bar_fig.update_traces(marker_color = 'rgba(0,0,250, 0.5)',
                        marker_line_width = 0,
                        selector=dict(type="bar"))
    return bar_fig


@app.callback(
    Output("result", "value"),
    [
        State('pass_count', 'value'),
        State('ml_pickup_hr', 'value'),
        State('ml_dropoff_hr', 'value'),
        State('my_trip_dist', 'value'),
        State('lat', 'value'),
        State('lon', 'value'),
        Input('predict', 'n_clicks')
    ],
)
def cb_render(pass_count, ml_pickup_hr, ml_dropoff_hr, my_trip_dist, address, zipcode, predict):
    if (predict):
        print(predict, pass_count, ml_pickup_hr, ml_dropoff_hr, my_trip_dist, address, zipcode)
        list_reg = []
        geolocator = Nominatim(user_agent="mp003")
        address = address + ", " + zipcode
        location = geolocator.geocode(address)

        list_reg.append(pass_count)
        list_reg.append(my_trip_dist)
        list_reg.append(location.latitude)
        list_reg.append(location.latitude)
        list_reg.append(ml_pickup_hr)
        list_reg.append(ml_dropoff_hr)
        reg = []
        reg.append(list_reg)
        print(list_reg)
        filename = 'finalized_model.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        y_pred = loaded_model.predict(reg)
        result = np.round(y_pred, 2)
        string = '$' + str(result[0])
        return string
    else: 
        string = str(0.00)
        string = '$' + string
        return string
    

if __name__=='__main__':
    app.run_server(debug=True, port=8000)
    app.config.suppress_callback_exceptions=True