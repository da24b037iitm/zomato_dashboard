#!/usr/bin/env python
# coding: utf-8

# # Zomato Restaurants — 3‑Page Dash App (Overview • Prediction • Details)
# 
# This notebook builds a single **Plotly Dash** app with a toggle (tabs) for three pages:
# 
# 1. **Overview (All Restaurants Analysis)** — KPIs, map, cost vs rating, cuisine breakdown, heatmap.
# 2. **Prediction (New Restaurant)** — input rating/cost/votes → rule‑based success/neutral/fail signal.
# 3. **Details (Restaurant Profile)** — name, address, cuisine, cost for two, rating, quick comparisons.
# 
# **Modules used:** `dash`, `dash_bootstrap_components`, `pandas`, `numpy`, `plotly.express`, `plotly.graph_objects`, plus stdlib.
# Provide a CSV named **`zomato_bengaluru.csv`** with columns:
# `name,address,city,location,cuisines,rate,votes,approx_cost(for two people),online_order,table_booking,lat,lon`.
# If the file isn't found, the notebook will generate a small synthetic dataset so the app runs.
# 

# In[95]:


import pandas as pd
import numpy as np
from datetime import datetime
import json, time
# from geopy.geocoders import Nominatim

import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc


# In[96]:


# Load data (expects 'zomato_bengaluru.csv'). If missing, create a demo dataset.
CSV_PATH = 'zomato_with_latlon.csv'

def load_or_make():
    df = pd.read_csv(CSV_PATH)
    # except Exception as e:
    #     rng = np.random.default_rng(42)
    #     areas = ['Koramangala','Indiranagar','Whitefield','HSR','BTM','JP Nagar']
    #     cuisines = ['North Indian','South Indian','Chinese','Fast Food','Italian','Cafe']
    #     n = 500
    #     df = pd.DataFrame({
    #         'name': [f'Restaurant {i+1}' for i in range(n)],
    #         'address': [f"{rng.integers(1,200)} Main Rd, {rng.choice(areas)}" for _ in range(n)],
    #         'city': ['Bengaluru']*n,
    #         'area': rng.choice(areas, size=n),
    #         'cuisines': rng.choice(cuisines, size=n),
    #         'rate': np.round(rng.uniform(2.2, 4.8, size=n), 1),
    #         'votes': rng.integers(5, 1500, size=n),
    #         'approx_cost(for two people)': rng.integers(200, 2000, size=n),
    #         'online_order': rng.choice(['Yes','No'], size=n, p=[0.7,0.3]),
    #         'table_booking': rng.choice(['Yes','No'], size=n, p=[0.4,0.6]),
    #         'lat': 12.97 + rng.normal(0, 0.03, size=n),
    #         'lon': 77.59 + rng.normal(0, 0.03, size=n),
    #     })
    # df['rating_bucket'] = pd.cut(df['rate'], bins=[0,2.5,3.5,4.0,5.0], labels=['<2.5','2.5–3.5','3.5–4.0','4.0+'])
    # df['cost_cat'] = pd.cut(df['approx_cost(for two people)'], bins=[0,500,1000,10000], labels=['Low','Mid','High'])
    return df
df = load_or_make()



# In[97]:


df.head()


# In[98]:


#clumns of df
df.columns


# In[99]:


df['rate']


# In[100]:


type(df['rate'][1])


# In[101]:


df.drop(df[df['rate']=='NEW'].index, inplace=True)
df.drop(df[df['rate']=='-'].index, inplace=True)
df.dropna(subset=['rate'], inplace=True)


# In[102]:


# df['rate'] = df['rate'].str.replace(r'\s*/5','',regex=True).astype(float)


# In[103]:


# df['rate']


# In[104]:


# df.dropna(inplace=True)


# In[105]:


# df.isnull().sum()


# In[106]:


# drop duplicated based on address
# df.drop_duplicates(subset=['address'], inplace=True)


# In[107]:


# df.info()


# In[108]:


# df['location']


# In[109]:


# # modifying online_order and table_booking columninto binary data
# # df['online_order'] = df['online_order'].map({'Yes':1,'No':0})
# df['book_table'] = df['book_table'].map({'Yes':1,'No':0})


# In[110]:


# df['approx_cost(for two people)'] = df['approx_cost(for two people)'].str.replace(r'[^0-9]*','',regex=True).astype(float)


# In[111]:


TOTAL = len(df)
AVG_RATING = float(np.round(df['rate'].mean(), 2))
AVG_COST = (df['approx_cost(for two people)'].mean())
PCT_4PLUS = (np.round((df['rate']>=4.0).mean()*100, 1))
PCT_4PLUS


# In[112]:


AVG_RATING


# In[113]:


AVG_COST


# In[114]:


# import re, time, logging
# from typing import Optional, Tuple
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from webdriver_manager.chrome import ChromeDriverManager
# from selenium.webdriver.common.by import By
# from selenium.webdriver.common.keys import Keys
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC

# logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# COORD_RE = re.compile(r'@(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),')

# def _extract(url: str):
#     m = COORD_RE.search(url)
#     return (float(m.group(1)), float(m.group(2))) if m else None

# def make_driver(headless: bool=True) -> webdriver.Chrome:
#     opts = webdriver.ChromeOptions()
#     if headless:
#         opts.add_argument("--headless=new")
#     opts.add_argument("--no-sandbox")
#     opts.add_argument("--disable-dev-shm-usage")
#     opts.add_argument("--window-size=1200,900")
#     # help in some environments:
#     opts.add_argument("--disable-gpu")
#     opts.add_argument("--disable-blink-features=AutomationControlled")
#     driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
#     # Hard caps so it can't hang forever:
#     driver.set_page_load_timeout(30)
#     driver.set_script_timeout(30)
#     return driver

# def _handle_consent(driver, wait):
#     """Dismiss Google's cookie/consent overlays if present."""
#     try:
#         # consent buttons vary by locale; try a few common selectors/texts
#         # 1) standard button
#         btn = wait.until(EC.element_to_be_clickable((
#             By.XPATH, "//button[.='I agree' or .='Agree' or .='Accept all' or .='Accept all cookies']"
#         )))
#         btn.click()
#         time.sleep(1)
#         return
#     except Exception:
#         pass
#     # fallback: look for generic consent shadow/iframe patterns
#     try:
#         # Sometimes an iframe is used
#         iframes = driver.find_elements(By.TAG_NAME, "iframe")
#         for f in iframes:
#             try:
#                 driver.switch_to.frame(f)
#                 btn = WebDriverWait(driver, 2).until(EC.element_to_be_clickable((
#                     By.XPATH, "//button[.='I agree' or .='Agree' or .='Accept all' or .='Accept all cookies']"
#                 )))
#                 btn.click()
#                 driver.switch_to.default_content()
#                 time.sleep(1)
#                 return
#             except Exception:
#                 driver.switch_to.default_content()
#     except Exception:
#         pass

# def get_coords_from_google_maps(query: str, total_timeout: int = 40) -> Optional[Tuple[float, float]]:
#     """Robust: times out, handles consent, retries."""
#     t0 = time.time()
#     driver = make_driver(headless=True)
#     try:
#         from urllib.parse import quote_plus
#         search_url = f"https://www.google.com/maps/search/?api=1&query={quote_plus(query)}"
#         logging.info("Opening %s", search_url)
#         driver.get(search_url)

#         wait = WebDriverWait(driver, 15)
#         _handle_consent(driver, wait)

#         # Wait for searchbox (page loaded)
#         try:
#             wait.until(EC.presence_of_element_located((By.ID, "searchboxinput")))
#         except Exception:
#             logging.info("searchboxinput not found; continuing anyway")

#         # Try immediate URL parse
#         coords = _extract(driver.current_url)
#         if coords:
#             return coords

#         # Re-trigger search to force navigation that writes @lat,lng
#         try:
#             box = driver.find_element(By.ID, "searchboxinput")
#             box.send_keys(Keys.ENTER)
#         except Exception:
#             pass

#         # Poll the URL for a few seconds (bounded)
#         end_poll = min(time.time() + 12, t0 + total_timeout - 5)
#         while time.time() < end_poll:
#             coords = _extract(driver.current_url)
#             if coords:
#                 return coords
#             time.sleep(0.5)

#         # Click first result in the left panel (if available)
#         try:
#             first = WebDriverWait(driver, 8).until(
#                 EC.element_to_be_clickable((By.CSS_SELECTOR, 'div[role="article"]'))
#             )
#             first.click()
#             WebDriverWait(driver, 10).until(lambda d: _extract(d.current_url) is not None)
#             coords = _extract(driver.current_url)
#             if coords:
#                 return coords
#         except Exception:
#             pass

#         return None
#     finally:
#         driver.quit()


# In[115]:


# import re, time
# from urllib.parse import quote_plus
# from selenium import webdriver
# from selenium.webdriver.firefox.options import Options as FirefoxOptions

# COORD_RE = re.compile(r'@(-?\d+\.\d+),(-?\d+\.\d+)')

# def get_coords_firefox(place: str, timeout: int = 25, headless: bool = True):
#     opts = FirefoxOptions()
#     opts.headless = headless

#     driver = webdriver.Firefox(options=opts)  # Selenium Manager fetches geckodriver
#     driver.set_page_load_timeout(timeout)

#     try:
#         url = f"https://www.google.com/maps/place/{quote_plus(place)}?hl=en&gl=IN"
#         driver.get(url)

#         # Poll the URL for @lat,lng up to timeout seconds
#         end = time.time() + timeout
#         while time.time() < end:
#             m = COORD_RE.search(driver.current_url)
#             if m:
#                 lat, lng = map(float, m.groups())
#                 return lat, lng
#             time.sleep(0)
#         return None, None
#     finally:
#         driver.quit()


# In[116]:


# locations=(df['location'].unique())
# coords={}


# In[ ]:





# In[ ]:





# In[117]:


# for i in locations:
#     coord=get_coords_firefox(i+', Bengaluru, Karnataka, India')
#     coords[i]= [coord[0], coord[1]]
#     print(f"Geocoded {i}: {coord}")


# In[118]:


# df['lat']=df['location'].map(lambda x: coords[x][0])
# df['lon']=df['location'].map(lambda x: coords[x][1])


# # In[119]:


# df.to_csv('zomato_with_latlon.csv', index=False)


# In[120]:


df["rating_bucket"] = pd.cut(df["rate"], bins=[0,2.5,3.5,4.0,5.0], labels=["<2.5","2.5–3.5","3.5–4.0","4.0+"])
df['cost_cat']  = pd.cut(df['approx_cost(for two people)'], bins=[0,500,1000,10000], labels=['Low','Mid','High'])


# In[121]:


def kpi_card(label, value, suffix=''):
    return dbc.Card(dbc.CardBody([
        html.Div(label, className='text-muted small'),
        html.H3(f"{value}{suffix}", className='mb-0')
    ]), className='shadow-sm')

def overview_figures(filtered):
    map_fig = px.scatter_mapbox(
        filtered, lat='lat', lon='lon', hover_name='name',
        hover_data={'rate': True, 'approx_cost(for two people)': True, 'location': True},
        color='rate', color_continuous_scale='Viridis', height=400, zoom=10
    )
    map_fig.update_layout(mapbox_style='carto-positron', margin=dict(l=0,r=0,t=0,b=0), coloraxis_colorbar=dict(title='Rating'))

    sc_fig = px.scatter(filtered, x='approx_cost(for two people)', y='rate',
                        color='cost_cat', hover_name='name',
                        labels={'approx_cost(for two people)':'Cost for Two (₹)','rate':'Rating'}, height=400)
    sc_fig.update_layout(margin=dict(l=0,r=0,t=10,b=0))

    top_cuis = (filtered.groupby('cuisines')['name'].count().sort_values(ascending=False).head(12).reset_index(name='count'))
    cuis_fig = px.bar(top_cuis, x='count', y='cuisines', orientation='h', height=400,
                      labels={'count':'Restaurants','cuisines':'Cuisine'})
    cuis_fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), yaxis={'categoryorder':'total ascending'})

    heat = filtered.pivot_table(values='rate', index='cost_cat', columns='location', aggfunc='mean')
    heat = heat.reindex(index=['Low','Mid','High'])
    heat_fig = go.Figure(data=go.Heatmap(z=heat.values, x=list(heat.columns), y=list(heat.index), coloraxis='coloraxis'))
    heat_fig.update_layout(height=400, margin=dict(l=0,r=0,t=10,b=0), coloraxis={'colorscale':'Blues'}, xaxis_title='Area', yaxis_title='Cost Category')
    return map_fig, sc_fig, cuis_fig, heat_fig

def details_panel(row, df_all):
    city_avg = df_all['rate'].mean()
    cuis_avg = df_all.loc[df_all['cuisines']==row['cuisines'], 'rate'].mean()
    bars = pd.DataFrame({'Metric':['Restaurant','City Avg','Cuisine Avg'], 'Rating':[row['rate'], city_avg, cuis_avg]})
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


# In[122]:


# Build the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

area_options = [{'label': a, 'value': a} for a in sorted(df['location'].dropna().unique())]
cuisine_options = [{'label': c, 'value': c} for c in sorted(df['cuisines'].dropna().unique())]

controls = dbc.Card(
    dbc.CardBody([
        html.Div('Filters', className='fw-bold mb-2'),
        dbc.Row([
            dbc.Col([dcc.Dropdown(options=area_options, id='area_dd', placeholder='Area', multi=True)], md=6),
            dbc.Col([dcc.Dropdown(options=cuisine_options, id='cuisine_dd', placeholder='Cuisine', multi=True)], md=6),
        ], className='gy-2'),
        dbc.Row([
            dbc.Col([dcc.RangeSlider(1.0, 5.0, 0.1, value=[2.5,5.0], id='rating_rs',
                                     tooltip={'placement':'bottom','always_visible':False})], md=12)
        ], className='mt-3'),
    ]),
    className='mb-3 shadow-sm'
)

tabs = dbc.Tabs([
    dbc.Tab(label='Overview', tab_id='tab-overview'),
    dbc.Tab(label='Prediction (New)', tab_id='tab-predict'),
    dbc.Tab(label='Details', tab_id='tab-details')
], id='tabs', active_tab='tab-overview', className='mb-3')

overview_layout = dbc.Container([
    dbc.Row([
        dbc.Col(kpi_card('Total Restaurants', f'{TOTAL}'), md=3),
        dbc.Col(kpi_card('Average Rating', AVG_RATING), md=3),
        dbc.Col(kpi_card('Average Cost for Two (₹)', AVG_COST), md=3),
        dbc.Col(kpi_card('% Rated 4+', f'{PCT_4PLUS}', suffix='%'), md=3),
    ], className='g-3 mb-1'),
    dbc.Row([dbc.Col(controls, md=12)], className='g-2'),
    dbc.Row([
        dbc.Col(dcc.Graph(id='map_fig'), md=6),
        dbc.Col(dcc.Graph(id='sc_fig'), md=6),
    ], className='g-3'),
    dbc.Row([
        dbc.Col(dcc.Graph(id='cuis_fig'), md=6),
        dbc.Col(dcc.Graph(id='heat_fig'), md=6),
    ], className='g-3'),
    dbc.Row([
        dbc.Col(dcc.Dropdown(id='restaurant_picker', placeholder='Select a restaurant for Details/Predict',
                             options=[{'label':n,'value':n} for n in sorted(df['name'].unique())]), md=12)
    ], className='g-3')
], fluid=True)

predict_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card(dbc.CardBody([
                html.Div('Inputs', className='fw-bold mb-2'),
                dbc.Row([dbc.Col(dcc.Slider(1.0,5.0,0.1, value=3.8, id='in_rating', tooltip={'placement':'bottom'}), md=12)], className='mb-3'),
                dbc.Row([
                    dbc.Col(dbc.Input(type='number', id='in_cost', value=500, min=100, step=50), md=6),
                    dbc.Col(dbc.Input(type='number', id='in_votes', value=250, min=0, step=10), md=6),
                ], className='gy-2'),
                html.Div(id='pred_result', className='mt-3'),
            ]), className='shadow-sm')
        ], md=4),
        dbc.Col([
            dbc.Card(dbc.CardBody([
                html.Div('Input vs Typical Ranges (City)', className='fw-bold mb-2'),
                dcc.Graph(id='predict_bars')
            ]), className='shadow-sm')
        ], md=8),
    ], className='g-3'),
], fluid=True)

details_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card(dbc.CardBody([
                html.H4(id='d_name'),
                html.Div(id='d_address', className='text-muted'),
                html.Hr(),
                html.Div(id='d_meta'),
            ]), className='shadow-sm')
        ], md=5),
        dbc.Col([
            dbc.Card(dbc.CardBody([
                html.Div('Rating vs Averages', className='fw-bold mb-2'),
                dcc.Graph(id='d_compare')
            ]), className='shadow-sm'),
            dbc.Card(dbc.CardBody([
                html.Div('Snapshot', className='fw-bold mb-2'),
                dcc.Graph(id='d_small_hist')
            ]), className='shadow-sm mt-3'),
        ], md=7),
    ], className='g-3'),
    dbc.Row([
        dbc.Col(dbc.Alert('Pick a restaurant from the Overview or the dropdown below.', color='info'), md=12),
        dbc.Col(dcc.Dropdown(id='d_picker', options=[{'label':n,'value':n} for n in sorted(df['name'].unique())], placeholder='Select restaurant'), md=6)
    ], className='g-3')
], fluid=True)

app.layout = dbc.Container([
    html.H2('Bengaluru Restaurants Dashboard'),
    tabs,
    html.Div(id='page_content')
], fluid=True)


# In[123]:


df


# In[124]:


df.columns


# In[125]:


# Render tab content
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
    if area_vals:
        f = f[f['location'].isin(area_vals)]
    if cuisine_vals:
        f = f[f['cuisines'].isin(cuisine_vals)]
    if rating_range:
        lo, hi = rating_range
        f = f[(f['rate']>=lo) & (f['rate']<=hi)]
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
    bars = pd.DataFrame({'Metric':['Rating','Votes','Cost'], 'Input':[float(in_rating), float(in_votes), float(in_cost)], 'City Avg':[df['rate'].groupby['city'].mean(), df['votes'].mean(), df['approx_cost(for two people)'].groupby['city'].mean()]})
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
        empty_fig = go.Figure(); empty_fig.update_layout(height=80, margin=dict(l=0,r=0,t=0,b=0))
        return '', '', '', empty_fig, empty_fig, None
    row = df[df['name']==chosen].iloc[0]
    name = row['name']
    address = f"{row.get('address','')} — {row.get('location','')}, {row.get('city','')}"
    meta = html.Div([
        html.Div(f"Cuisine: {row['cuisines']}"),
        html.Div(f"Cost for Two: ₹{int(row['approx_cost(for two people)'])}"),
        html.Div(f"Rating: {row['rate']}  | Votes: {int(row['votes'])}"),
        html.Div(f"Online Order: {row.get('online_order','N/A')}  | Table Booking: {row.get('table_booking','N/A')}")
    ])
    comp_fig = details_panel(row, df)
    hist = px.histogram(df, x='rate', nbins=20, height=180)
    hist.add_vline(x=row['rate'], line_dash='dash')
    hist.update_layout(margin=dict(l=0,r=0,t=0,b=0))
    return name, address, meta, comp_fig, hist, chosen


# In[126]:


# Run the app
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)


# In[ ]:





# In[ ]:




