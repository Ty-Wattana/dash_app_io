import dash
import dash_cytoscape as cyto
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import dash_reusable_components as drc

import os, requests, re
import json
import numpy as np
from datetime import datetime
import time
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import networkx as nx

import plotly.express as px

def MAP(x,level):
  return nodes[nodes['id'] == x][f'id{level}'].reset_index(drop=True)[0]

def get_start_level(io_deepest,selected_level="16"):
  io_newLevel = io_deepest.copy()

  # convert to selected level

  if selected_level == "180":
    return io_newLevel
  else:
    io_newLevel[f'COLUMN_{selected_level}'] = io_newLevel['COLUMN'].apply(MAP, level=selected_level)
    io_newLevel[f'ROW_{selected_level}'] = io_newLevel['ROW'].apply(MAP, level=selected_level)
  

  io_newLevel = io_newLevel[[f'COLUMN_{selected_level}',f'ROW_{selected_level}','weight']]
  io_newLevel = io_newLevel.groupby(by=[f'COLUMN_{selected_level}',f'ROW_{selected_level}'],as_index=False).sum()
  io_newLevel['Buyer'] = io_newLevel[f'COLUMN_{selected_level}']
  io_newLevel['Seller'] = io_newLevel[f'ROW_{selected_level}']

  io_newLevel.drop_duplicates(inplace=True)

  scaler = MinMaxScaler((1,5))
  scaler_outlier = MinMaxScaler((5,6))

  # filter outlier
  Q1 = io_newLevel['weight'].quantile(0.25)
  Q3 = io_newLevel['weight'].quantile(0.75)
  IQR = Q3 - Q1    #IQR is interquartile range. 

  filter = (io_newLevel['weight'] >= Q1 - 1.5 * IQR) & (io_newLevel['weight'] <= Q3 + 1.5 *IQR)
  io_newLevel["weight_norm"] = 0

  io_newLevel.loc[filter,"weight_norm"] = scaler.fit_transform(io_newLevel.loc[filter][["weight"]])
  io_newLevel.loc[~filter,"weight_norm"] = scaler_outlier.fit_transform(io_newLevel.loc[~filter][["weight"]])

  return io_newLevel

# get data
io_2015 = pd.read_excel('./io_data/IO Table 2010.xlsx', dtype={'ROW':'object','COLUMN':'object'})
nodes = pd.read_excel('./io_data/node.xlsx', dtype={'id':'object', 'id58':'object','id26':'object','id16':'object'})

irr_node = ['190','201','202','203','204','209','210','301','302','303','304',
            '305','306','309','310','401','402','403','404','409','501','502', '503',
            '509','600','700']
           
# preprocessing data
nodes = nodes[~nodes['id'].isin(irr_node)]
io_2015 = io_2015[~io_2015['ROW'].isin(irr_node)]
io_2015 = io_2015[~io_2015['COLUMN'].isin(irr_node)]

io_2015 = io_2015[['COLUMN','ROW','PURCHASER']]
io_2015.rename(columns = {'PURCHASER':'weight'},inplace=True)
io_2015['Buyer'] = io_2015['COLUMN']
io_2015['Seller'] = io_2015['ROW']

scaler = MinMaxScaler((1,5))
scaler_outlier = MinMaxScaler((5,6))

# filter outlier
Q1 = io_2015['weight'].quantile(0.25)
Q3 = io_2015['weight'].quantile(0.75)
IQR = Q3 - Q1    #IQR is interquartile range. 

filter = (io_2015['weight'] >= Q1 - 1.5 * IQR) & (io_2015['weight'] <= Q3 + 1.5 *IQR)

io_2015.loc[filter,"weight_norm"] = scaler.fit_transform(io_2015.loc[filter][["weight"]])
io_2015.loc[~filter,"weight_norm"] = scaler_outlier.fit_transform(io_2015.loc[~filter][["weight"]])

nodes['id58'] = nodes['id58'] + "_58"
nodes['id26'] = nodes['id26'] + "_26"
nodes['id16'] = nodes['id16'] + "_16"

# convert to 16

io_2015_16 = io_2015.copy()
io_2015_16['COLUMN_16'] = io_2015_16['COLUMN'].apply(MAP, level="16")
io_2015_16['ROW_16'] = io_2015_16['ROW'].apply(MAP, level="16")

io_2015_16 = io_2015_16[['COLUMN_16','ROW_16','weight']]
io_2015_16 = io_2015_16.groupby(by=['COLUMN_16','ROW_16'],as_index=False).sum()
io_2015_16['Buyer'] = io_2015_16['COLUMN_16']
io_2015_16['Seller'] = io_2015_16['ROW_16']
io_2015_16


scaler = MinMaxScaler((1,5))
scaler_outlier = MinMaxScaler((5,6))

# filter outlier
Q1 = io_2015_16['weight'].quantile(0.25)
Q3 = io_2015_16['weight'].quantile(0.75)
IQR = Q3 - Q1    #IQR is interquartile range. 

filter = (io_2015_16['weight'] >= Q1 - 1.5 * IQR) & (io_2015_16['weight'] <= Q3 + 1.5 *IQR)
io_2015_16["weight_norm"] = 0

io_2015_16.loc[filter,"weight_norm"] = scaler.fit_transform(io_2015_16.loc[filter][["weight"]])
io_2015_16.loc[~filter,"weight_norm"] = scaler_outlier.fit_transform(io_2015_16.loc[~filter][["weight"]])

# get graph element
G = nx.from_pandas_edgelist(io_2015_16, 
                            'Buyer', 
                            'Seller', 
                            ['weight','weight_norm'],
                            create_using=nx.DiGraph())

# get dash cytoscape elements
cyto_elements = []
edge_weight = nx.get_edge_attributes(G,"weight_norm")
amount = nx.get_edge_attributes(G,"weight")

for index,row in nodes.iterrows():
  temp_dict = {}
  temp_dict['data'] = {}
  temp_dict['data']['id'] = row['id16']
  temp_dict['data']['label'] = row['name16']

  cyto_elements.append(temp_dict)

for edge in G.edges:
  temp_dict = {}

  temp_dict['data'] = {}
  temp_dict['data']['source'] = edge[0]
  temp_dict['data']['target'] = edge[1]
  temp_dict['data']['weight'] = edge_weight[edge]
  temp_dict['data']['amount'] = amount[edge]

  cyto_elements.append(temp_dict)

default_stylesheet = [
    {
        "selector": 'node',
        'style': {
            "label": "data(label)",
            "color": "black",
            "text-opacity": 0.65,
            "font-size": 10,
        }
    },
    {
        "selector": 'edge',
        'style': {
            "curve-style": "bezier",
            "opacity": 0.3,
            'target-arrow-shape': 'vee',
            "width":"data(weight)"
        }
    },
]

styles = {
    'json-output': {
        'overflow-y': 'scroll',
        'height': 'calc(50% - 25px)',
        'border': 'thin lightgrey solid'
    },
    'tab': {
        'height': 'calc(98vh - 105px)'
    }
}


# dash app

app = Dash(__name__)
application = app.server
app.layout = html.Div([dcc.Tabs([
    dcc.Tab(label='Spatial Visualization', children=[
    html.H1("Input-Output Economic Network (2015)",style={"text-align":"center"}),

    html.Div(className='network-stat', children=[
        html.H3("Network Diameter: ",style={"text-align":"center","border":"2px black solid","padding":"10px",'width':'20%',
                                            "margin-left": "15px"}),
        html.H3("Network Density: ",style={"text-align":"center","border":"2px black solid","padding":"10px",'width':'20%',
                                            "margin-left": "15px"}),
        html.H3("Average Shortest Path: ",style={"text-align":"center","border":"2px black solid","padding":"10px",'width':'20%',
                                                 "margin-left": "15px"}),
        html.H3("Average Cluster Coeficient: ",style={"text-align":"center","border":"2px black solid","padding":"10px",'width':'20%',
                                                      "margin-left": "15px"})
    ],style={'display':"flex"}),

    html.Div(className='network-control-panel', children=[
        cyto.Cytoscape(
            id='IO-network',
            elements=cyto_elements,
            style={
                'height': '500px',
                'width': '70%'
                },
            boxSelectionEnabled = True
            ),
        html.Div(className='four columns', children=[
            dcc.Tabs(id='tabs', children=[
              dcc.Tab(label='Control Panel', children=[
                  drc.NamedDropdown(
                      name='Layout',
                      id='dropdown-layout',
                      options=drc.DropdownOptionsList(
                          'random',
                          'grid',
                          'circle',
                          'concentric',
                          'breadthfirst',
                          'cose'
                      ),
                      value='circle',
                      clearable=False
                  ),
                  drc.NamedDropdown(
                      name='Level',
                      id='dropdown-level',
                      options=drc.DropdownOptionsList(
                          '16',
                          '26',
                          '58',
                          '180'
                      ),
                      value='16',
                      clearable=False
                  ),
              ]),
          ]),
      ],style={'width': '30%'}
      ) 
    ],style = {'display' : 'flex'}),

    html.Div(className='nodeInfo-sankey', children=[
        dcc.Graph(style={'width':'50%'}),
        dcc.Graph(style={'width':'50%'}),
    ],style={'display':"flex"}),

    html.Div(className='adj-metrix', children=[
        dcc.Graph(style={'width':'100%'})
    ],style={'display':"flex"}),

    html.Div(className='centrailities', children=[
        dcc.Graph(style={'width':'25%'}),
        dcc.Graph(style={'width':'25%'}),
        dcc.Graph(style={'width':'25%'}),
        dcc.Graph(style={'width':'25%'}),
    ],style={'display':"flex"}),

    html.Div(className='cluster', children=[
        dcc.Graph(style={'width':'100%'}),
    ],style={'display':"flex"}),

    ]),
    dcc.Tab(label='Temporal Visualization', children=[
        html.H1("Under Development",style={"text-align":"center"})
    ])
  ])
])


@app.callback(Output('IO-network', 'layout'),
              [Input('dropdown-layout', 'value')])
def update_cytoscape_layout(layout):
    return {'name': layout}


# @app.callback(Output('IO-network', 'stylesheet'),
#               Output('IO-network', 'elements'),
#               [Input('IO-network', 'tapNode')],
#               [State('IO-network', 'elements')])

# def generate_stylesheet(node,elements):
#     if not node:
#         return default_stylesheet,elements

#     if node['data'].get('selected') == 1:
#       idx = elements.index({"data":node['data']})
#       elements[idx]['data']['selected'] = 0
#     else:
#       idx = elements.index({"data":node['data']})
#       elements[idx]['data']['selected'] = 1
    
#     selected_nodes = [element.get('data').get('id') for element in elements if element.get('data').get('selected') == 1]
#     all_edges = [element.get('data') for element in elements if "source" in element['data']]

#     stylesheet = [{
#         "selector": 'node',
#         'style': {
#             "label": "data(label)",
#             "color": "black",
#             "text-opacity": 0.65,
#             "font-size": 10,
#             'shape': "ellipse"
#         }
#     }, {
#         'selector': 'edge',
#         'style': {
#             'opacity': 0.3,
#             "curve-style": "bezier",
#             "width":"data(weight)"
#         }
#     }, {
#         "selector": '[selected > 0]',
#         "style": {
#             'background-color': '#B10DC9',
#             "border-color": "purple",
#             "border-width": 2,
#             "border-opacity": 1,
#             "opacity": 1,

#             "label": "data(label)",
#             "color": "#B10DC9",
#             "text-opacity": 1,
#             "font-size": 12,
#             'z-index': 9999
#         }
#     }]
#     for edge in all_edges:
#         if edge['source'] in selected_nodes:#== node['data']['id']:
#             stylesheet.append({
#                 "selector": 'node[id = "{}"]'.format(edge['target']),
#                 "style": {
#                     'background-color': "red",
#                     "label": "data(label)",
#                     "color": "red",
#                     "text-opacity": 1,
#                     "font-size": 12,
#                     'opacity': 0.9
#                 }
#             })
#             stylesheet.append({
#                 "selector": 'edge[id= "{}"]'.format(edge['id']),
#                 "style": {
#                     "target-arrow-color": "red",
#                     "target-arrow-shape": "vee",
#                     "line-color": "red",
#                     'opacity': 0.9,
#                     'z-index': 5000,
#                     "width":"data(weight)"
#                 }
#             })

#         if edge['target'] in selected_nodes:#== node['data']['id']:
#             stylesheet.append({
#                 "selector": 'node[id = "{}"]'.format(edge['source']),
#                 "style": {
#                     'background-color': "green",
#                     "label": "data(label)",
#                     "color": "green",
#                     "text-opacity": 1,
#                     "font-size": 12,
#                     'opacity': 0.9,
#                     'z-index': 9999
#                 }
#             })
#             stylesheet.append({
#                 "selector": 'edge[id= "{}"]'.format(edge['id']),
#                 "style": {
#                     "target-arrow-color": "green",
#                     "target-arrow-shape": "vee",
#                     "line-color": "green",
#                     'opacity': 1,
#                     'z-index': 5000,
#                     "width":"data(weight)"
#                 }
#             })

#     return stylesheet,elements

@app.callback(Output('IO-network', 'stylesheet'),
              Output('IO-network', 'elements'),
              [Input('IO-network', 'tapNode')],
              [Input('dropdown-level', 'value')],
              [State('IO-network', 'elements')])

def expandElement(node,start_level,elements):
    if not node: # no node selected
        # reconstruct cyto element with start level

        io_start = get_start_level(io_2015,start_level)

        G_start = nx.from_pandas_edgelist(io_start, 
                                'Buyer', 
                                'Seller', 
                                ['weight','weight_norm'],
                                create_using=nx.DiGraph())

        # get dash cytoscape elements
        cyto_start = []
        edge_weight = nx.get_edge_attributes(G_start,"weight_norm")
        amount = nx.get_edge_attributes(G_start,"weight")

        if start_level != "16" and start_level != "26" and start_level != "58":

            for index,row in nodes.iterrows():
                temp_dict = {}
                temp_dict['data'] = {}
                temp_dict['data']['id'] = row["id"]
                temp_dict['data']['label'] = row['name']
                temp_dict['data']['level'] = start_level
                cyto_start.append(temp_dict)

        else:
            nodes_cyto = nodes.groupby([f'id{start_level}',f'name{start_level}']).mean().reset_index()
            for index,row in nodes_cyto.iterrows():
                temp_dict = {}
                temp_dict['data'] = {}
                temp_dict['data']['id'] = row[f'id{start_level}']
                temp_dict['data']['label'] = row[f'name{start_level}']
                temp_dict['data']['level'] = start_level
                cyto_start.append(temp_dict)

        for edge in G_start.edges:
            temp_dict = {}

            temp_dict['data'] = {}
            temp_dict['data']['source'] = edge[0]
            temp_dict['data']['target'] = edge[1]
            temp_dict['data']['weight'] = edge_weight[edge]
            temp_dict['data']['amount'] = amount[edge]

            cyto_start.append(temp_dict)

        return default_stylesheet,cyto_start

    selected_id = node["data"]["id"]
    selected_level = node["data"]["level"]

    expandedID_deepest = list(nodes.loc[nodes[f"id{selected_level}"] == selected_id,"id"])
    
    io_selected = io_2015.copy()
    node_selected = nodes.copy()

    # delete selected node and associated links from cytoscape element
    new_cytoElement = elements

    for c in new_cytoElement: # loop to delete
      if selected_id in c['data'].values():
        new_cytoElement.remove(c)

    # get selected node deeper level data
    if selected_level == "16":
      io_selected = io_selected[(io_selected["COLUMN"].isin(expandedID_deepest)) | (io_selected["ROW"].isin(expandedID_deepest))]

      # convert non-selected nodes
      io_selected.loc[~(io_selected["COLUMN"].isin(expandedID_deepest)),"COLUMN"] = io_selected.loc[~(io_selected["COLUMN"].isin(expandedID_deepest)),"COLUMN"].apply(MAP, level="16")
      io_selected.loc[~(io_selected["ROW"].isin(expandedID_deepest)),"ROW"] = io_selected.loc[~(io_selected["ROW"].isin(expandedID_deepest)),"ROW"].apply(MAP, level="16")

      # convert selected nodes

      io_selected.loc[(io_selected["COLUMN"].isin(expandedID_deepest)),"COLUMN"] = io_selected.loc[(io_selected["COLUMN"].isin(expandedID_deepest)),"COLUMN"].apply(MAP, level="26")
      io_selected.loc[(io_selected["ROW"].isin(expandedID_deepest)),"ROW"] = io_selected.loc[(io_selected["ROW"].isin(expandedID_deepest)),"ROW"].apply(MAP, level="26")

      node_selected = node_selected[node_selected["id"].isin(expandedID_deepest)]
      node_selected = node_selected[["id26","name26"]]
      node_selected.rename(columns={"id26":"id","name26":"name"},inplace=True)
      node_selected.drop_duplicates(inplace=True)

    # elif selected_level == "26":


    # calculate edge weight
    io_selected = io_selected[['COLUMN','ROW','weight']]
    io_selected = io_selected.groupby(by=['COLUMN','ROW'],as_index=False).sum()
    io_selected['Buyer'] = io_selected['COLUMN']
    io_selected['Seller'] = io_selected['ROW']

    io_selected.drop_duplicates(inplace=True)

    scaler = MinMaxScaler((1,5))
    scaler_outlier = MinMaxScaler((5,6))

    Q1 = io_selected['weight'].quantile(0.25)
    Q3 = io_selected['weight'].quantile(0.75)
    IQR = Q3 - Q1    #IQR is interquartile range. 

    filter = (io_selected['weight'] >= Q1 - 1.5 * IQR) & (io_selected['weight'] <= Q3 + 1.5 *IQR)
    io_selected["weight_norm"] = 0

    io_selected.loc[filter,"weight_norm"] = scaler.fit_transform(io_selected.loc[filter][["weight"]])
    io_selected.loc[~filter,"weight_norm"] = scaler_outlier.fit_transform(io_selected.loc[~filter][["weight"]])

    # get nx object
    G_start = nx.from_pandas_edgelist(io_selected, 
                            'Buyer', 
                            'Seller', 
                            ['weight','weight_norm'],
                            create_using=nx.DiGraph())
        
    edge_weight = nx.get_edge_attributes(G_start,"weight_norm")
    amount = nx.get_edge_attributes(G_start,"weight")

    # append to old cyto element

    for index,row in node_selected.iterrows():
      temp_dict = {}
      temp_dict['data'] = {}
      temp_dict['data']['id'] = row['id']
      temp_dict['data']['label'] = row['name']

      new_cytoElement.append(temp_dict)

    for edge in G_start.edges:
      temp_dict = {}

      temp_dict['data'] = {}
      temp_dict['data']['source'] = edge[0]
      temp_dict['data']['target'] = edge[1]
      temp_dict['data']['weight'] = edge_weight[edge]
      temp_dict['data']['amount'] = amount[edge]

      new_cytoElement.append(temp_dict)
    
    return default_stylesheet,new_cytoElement
        


if __name__ == '__main__':
    app.run_server(debug=True)