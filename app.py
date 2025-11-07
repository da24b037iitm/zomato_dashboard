#!/usr/bin/env python
# coding: utf-8

# # 🍽️ Zomato Restaurants — 3‑Page Dash App (Self‑Contained Version)
# 
# This notebook builds a **Plotly Dash** app for **Bengaluru restaurants**, featuring:
# 
# 1. **Overview Page** — KPIs, map, scatter (Cost vs Rating), cuisine bars, heatmap.  
# 2. **Prediction Page** — Predict restaurant success (Success / Neutral / Fail).  
# 3. **Details Page** — View full details for any restaurant.
# 
# ### ✅ What's new in this version:
# - No external CSV file — synthetic Bengaluru dataset generated automatically.
# - Fixed all callback and layout issues.
# - Uses only your approved libraries (`dash`, `dash_bootstrap_components`, `pandas`, `numpy`, `plotly`).
# 
# Run the last cell to launch the interactive dashboard.
# 

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime
import json, time

import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc


# In[2]:


# --- Synthetic Bengaluru Restaurant Data ---
rng = np.random.default_rng(42)
areas = ["Koramangala", "Indiranagar", "Whitefield", "HSR", "BTM", "JP Nagar"]
cuisines = ["North Indian", "South Indian", "Chinese", "Fast Food", "Italian", "Cafe"]
n = 500

df = pd.DataFrame({
    "name": [f"Restaurant {i+1}" for i in range(n)],
    "address": [f"{rng.integers(1,200)} Main Rd, {rng.choice(areas)}" for _ in range(n)],
    "city": ["Bengaluru"] * n,
    "area": rng.choice(areas, size=n),
    "cuisines": rng.choice(cuisines, size=n),
    "avg_rating": np.round(rng.uniform(2.2, 4.8, size=n), 1),
    "votes": rng.integers(5, 1500, size=n),
    "cost_for_two": rng.integers(200, 2000, size=n),
    "online_order": rng.choice(["Yes", "No"], size=n, p=[0.7, 0.3]),
    "table_booking": rng.choice(["Yes", "No"], size=n, p=[0.4, 0.6]),
    "lat": 12.97 + rng.normal(0, 0.03, size=n),
    "lon": 77.59 + rng.normal(0, 0.03, size=n),
})

# Categorization
df["rating_bucket"] = pd.cut(df["avg_rating"], bins=[0,2.5,3.5,4.0,5.0], labels=["<2.5","2.5–3.5","3.5–4.0","4.0+"])
df["cost_cat"] = pd.cut(df["cost_for_two"], bins=[0,500,1000,10000], labels=["Low","Mid","High"])

# KPIs
TOTAL = len(df)
AVG_RATING = float(np.round(df["avg_rating"].mean(), 2))
AVG_COST = int(df["cost_for_two"].mean())
PCT_4PLUS = float(np.round((df["avg_rating"] >= 4.0).mean() * 100, 1))

print(f"✅ Generated {TOTAL} restaurants for Bengaluru — ready for dashboard.")


# In[3]:


def kpi_card(label, value, suffix=''):
    return dbc.Card(dbc.CardBody([
        html.Div(label, className='text-muted small'),
        html.H3(f"{value}{suffix}", className='mb-0')
    ]), className='shadow-sm')

def overview_figures(filtered):
    map_fig = px.scatter_mapbox(
        filtered, lat='lat', lon='lon', hover_name='name',
        hover_data={'avg_rating': True, 'cost_for_two': True, 'area': True},
        color='avg_rating', color_continuous_scale='Viridis', height=400, zoom=10
    )
    map_fig.update_layout(mapbox_style='carto-positron', margin=dict(l=0,r=0,t=0,b=0))

    sc_fig = px.scatter(filtered, x='cost_for_two', y='avg_rating',
                        color='cost_cat', hover_name='name',
                        labels={'cost_for_two':'Cost for Two (₹)','avg_rating':'Rating'}, height=400)
    sc_fig.update_layout(margin=dict(l=0,r=0,t=10,b=0))

    top_cuis = (filtered.groupby('cuisines')['name'].count().sort_values(ascending=False).head(12).reset_index(name='count'))
    cuis_fig = px.bar(top_cuis, x='count', y='cuisines', orientation='h', height=400,
                      labels={'count':'Restaurants','cuisines':'Cuisine'})
    cuis_fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), yaxis={'categoryorder':'total ascending'})

    heat = filtered.pivot_table(values='avg_rating', index='cost_cat', columns='area', aggfunc='mean')
    heat = heat.reindex(index=['Low','Mid','High'])
    heat_fig = go.Figure(data=go.Heatmap(z=heat.values, x=list(heat.columns), y=list(heat.index), colorscale='Blues'))
    heat_fig.update_layout(height=400, margin=dict(l=0,r=0,t=10,b=0), xaxis_title='Area', yaxis_title='Cost Category')
    return map_fig, sc_fig, cuis_fig, heat_fig

def details_panel(row, df_all):
    city_avg = df_all['avg_rating'].mean()
    cuis_avg = df_all.loc[df_all['cuisines']==row['cuisines'], 'avg_rating'].mean()
    bars = pd.DataFrame({'Metric':['Restaurant','City Avg','Cuisine Avg'], 'Rating':[row['avg_rating'], city_avg, cuis_avg]})
    bar_fig = px.bar(bars, x='Metric', y='Rating', range_y=[0,5], height=260)
    bar_fig.update_layout(margin=dict(l=0,r=0,t=0,b=0))
    return bar_fig

def success_label(input_rating, votes, cost):
    if (input_rating >= 4.0 and votes > 200 and cost >= 600):
        return 'Success', 'success'
    elif (input_rating < 3.0 or votes < 50):
        return 'Fail', 'danger'
    else:
        return 'Neutral', 'warning'


# In[4]:


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

area_options = [{'label': a, 'value': a} for a in sorted(df['area'].dropna().unique())]
cuisine_options = [{'label': c, 'value': c} for c in sorted(df['cuisines'].dropna().unique())]

# Tabs
tabs = dbc.Tabs([
    dbc.Tab(label='Overview', tab_id='tab-overview'),
    dbc.Tab(label='Prediction (New)', tab_id='tab-predict'),
    dbc.Tab(label='Details', tab_id='tab-details')
], id='tabs', active_tab='tab-overview', className='mb-3')

# Overview Layout
overview_layout = dbc.Container([
    dbc.Row([
        dbc.Col(kpi_card('Total Restaurants', f'{TOTAL}'), md=3),
        dbc.Col(kpi_card('Average Rating', AVG_RATING), md=3),
        dbc.Col(kpi_card('Average Cost for Two (₹)', AVG_COST), md=3),
        dbc.Col(kpi_card('% Rated 4+', f'{PCT_4PLUS}', suffix='%'), md=3),
    ], className='g-3 mb-1'),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div('Filters', className='fw-bold mb-2'),
            dbc.Row([
                dbc.Col(dcc.Dropdown(options=area_options, id='area_dd', placeholder='Area', multi=True), md=6),
                dbc.Col(dcc.Dropdown(options=cuisine_options, id='cuisine_dd', placeholder='Cuisine', multi=True), md=6)
            ], className='gy-2'),
            dbc.Row([
                dbc.Col(dcc.RangeSlider(1.0, 5.0, 0.1, value=[2.5,5.0], id='rating_rs', tooltip={'placement':'bottom'}), md=12)
            ], className='mt-3')
        ]), className='shadow-sm'), md=12)
    ], className='g-2'),
    dbc.Row([dbc.Col(dcc.Graph(id='map_fig'), md=6), dbc.Col(dcc.Graph(id='sc_fig'), md=6)], className='g-3'),
    dbc.Row([dbc.Col(dcc.Graph(id='cuis_fig'), md=6), dbc.Col(dcc.Graph(id='heat_fig'), md=6)], className='g-3'),
    dbc.Row([dbc.Col(dcc.Dropdown(id='restaurant_picker', placeholder='Select restaurant for Details', options=[{'label':n,'value':n} for n in sorted(df['name'].unique())]), md=12)], className='g-3')
], fluid=True)

# Prediction Layout
predict_layout = dbc.Container([
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div('Inputs', className='fw-bold mb-2'),
            dcc.Slider(1.0,5.0,0.1,value=3.8,id='in_rating',tooltip={'placement':'bottom'}),
            dbc.Row([
                dbc.Col(dbc.Input(type='number', id='in_cost', value=500, min=100, step=50), md=6),
                dbc.Col(dbc.Input(type='number', id='in_votes', value=250, min=0, step=10), md=6)
            ], className='gy-2 mt-2'),
            html.Div(id='pred_result', className='mt-3')
        ]), className='shadow-sm'), md=4),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div('Input vs City Averages', className='fw-bold mb-2'),
            dcc.Graph(id='predict_bars')
        ]), className='shadow-sm'), md=8)
    ], className='g-3')
], fluid=True)

# Details Layout
details_layout = dbc.Container([
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4(id='d_name'), html.Div(id='d_address', className='text-muted'), html.Hr(), html.Div(id='d_meta')
        ]), className='shadow-sm'), md=5),
        dbc.Col([
            dbc.Card(dbc.CardBody([html.Div('Rating vs Averages', className='fw-bold mb-2'), dcc.Graph(id='d_compare')]), className='shadow-sm'),
            dbc.Card(dbc.CardBody([html.Div('Snapshot', className='fw-bold mb-2'), dcc.Graph(id='d_small_hist')]), className='shadow-sm mt-3')
        ], md=7)
    ], className='g-3'),
    dbc.Row([dbc.Col(dcc.Dropdown(id='d_picker', options=[{'label':n,'value':n} for n in sorted(df['name'].unique())], placeholder='Select restaurant'), md=6)], className='g-3')
], fluid=True)

app.layout = dbc.Container([
    html.H2('Bengaluru Restaurants Dashboard'),
    tabs,
    html.Div(id='page_content')
], fluid=True)


# In[5]:


@app.callback(Output('page_content','children'), Input('tabs','active_tab'))
def render_tab(tab_id):
    if tab_id == 'tab-overview':
        return overview_layout
    elif tab_id == 'tab-predict':
        return predict_layout
    else:
        return details_layout

@app.callback(
    [Output('map_fig','figure'), Output('sc_fig','figure'), Output('cuis_fig','figure'), Output('heat_fig','figure'), Output('restaurant_picker','options')],
    [Input('area_dd','value'), Input('cuisine_dd','value'), Input('rating_rs','value')]
)
def update_overview(area_vals, cuisine_vals, rating_range):
    f = df.copy()
    if area_vals: f = f[f['area'].isin(area_vals)]
    if cuisine_vals: f = f[f['cuisines'].isin(cuisine_vals)]
    if rating_range: lo, hi = rating_range; f = f[(f['avg_rating']>=lo)&(f['avg_rating']<=hi)]
    map_fig, sc_fig, cuis_fig, heat_fig = overview_figures(f if len(f)>0 else df.head(0))
    opts = [{'label':n,'value':n} for n in sorted(f['name'].unique())] if len(f)>0 else []
    return map_fig, sc_fig, cuis_fig, heat_fig, opts

@app.callback(
    [Output('pred_result','children'), Output('predict_bars','figure')],
    [Input('in_rating','value'), Input('in_votes','value'), Input('in_cost','value')]
)
def compute_prediction(in_rating, in_votes, in_cost):
    label, color = success_label(float(in_rating), int(in_votes), int(in_cost))
    msg = dbc.Alert(f'Prediction: {label}', color=color, className='mb-0')
    bars = pd.DataFrame({'Metric':['Rating','Votes','Cost'], 'Input':[in_rating, in_votes, in_cost],
                         'City Avg':[df['avg_rating'].mean(), df['votes'].mean(), df['cost_for_two'].mean()]})
    fig = px.bar(bars.melt(id_vars=['Metric'], var_name='Type', value_name='Value'), x='Metric', y='Value', color='Type', barmode='group', height=320)
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), legend_title_text='')
    return msg, fig

@app.callback(
    [Output('d_name','children'), Output('d_address','children'), Output('d_meta','children'), Output('d_compare','figure'), Output('d_small_hist','figure'), Output('d_picker','value')],
    [Input('d_picker','value'), Input('restaurant_picker','value')]
)
def update_details(d_pick, from_overview):
    chosen = from_overview or d_pick
    if not chosen:
        empty = go.Figure(); empty.update_layout(height=100, margin=dict(l=0,r=0,t=0,b=0))
        return '', '', '', empty, empty, None
    row = df[df['name']==chosen].iloc[0]
    name, address = row['name'], f"{row['address']} — {row['area']}, {row['city']}"
    meta = html.Div([
        html.Div(f"Cuisine: {row['cuisines']}"),
        html.Div(f"Cost for Two: ₹{int(row['cost_for_two'])}"),
        html.Div(f"Rating: {row['avg_rating']}  | Votes: {int(row['votes'])}"),
        html.Div(f"Online Order: {row['online_order']}  | Table Booking: {row['table_booking']}")
    ])
    comp_fig = details_panel(row, df)
    hist = px.histogram(df, x='avg_rating', nbins=20, height=180); hist.add_vline(x=row['avg_rating'], line_dash='dash')
    hist.update_layout(margin=dict(l=0,r=0,t=0,b=0))
    return name, address, meta, comp_fig, hist, chosen


# In[6]:


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)

