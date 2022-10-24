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
import plotly.graph_objects as go

from utils.util import MAP, get_start_level, get_all_levels, get_edge_df_from_cyto, get_node_name, genSankey, empty_fig

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


############################################ dash app ##################################################

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
                      name='Starting Level',
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
                      name='Mode',
                      id='dropdown-mode',
                      options=drc.DropdownOptionsList(
                          'selection',
                          'expansion',
                      ),
                      value='selection',
                      clearable=False
                  ),
              ]),
          ]),
      ],style={'width': '30%'}
      ) 
    ],style = {'display' : 'flex'}),

    html.Div(className='nodeInfo-sankey', children=[
        dcc.Graph(style={'width':'100%'}, id="node-sanky"),
    ],style={'display':"flex"}),

    html.Div(className='nodeInfo-donut', children=[
        dcc.Graph(style={'width':'50%'}, id="node-donut-inflow"),
        dcc.Graph(style={'width':'50%'}, id="node-donut-outflow")
    ],style={'display':"flex"}),

    html.Div(className='overAllcentrailities', children=[
        dcc.Graph(style={'width':'100%'}, id="all-degree-centrality"),
    ],style={'display':"flex"}),

    html.Div(className='centrailities', children=[
        dcc.Graph(style={'width':'50%'}, id="in-degree-centrality"),
        dcc.Graph(style={'width':'50%'}, id="out-degree-centrality"),
    ],style={'display':"flex"}),

    html.Div(className='cluster', children=[
        dcc.Graph(style={'width':'100%'}),
    ],style={'display':"flex"}),

    ]),

    html.Div(className='adj-metrix', children=[
        dcc.Graph(style={'width':'100%'})
    ],style={'display':"flex"}),

    
    dcc.Tab(label='Temporal Visualization', children=[
        html.H1("Under Development",style={"text-align":"center"})
    ])
  ])
])


############################################ Call Back ###################################################


@app.callback(Output('IO-network', 'layout'),
              [Input('dropdown-layout', 'value')])
def update_cytoscape_layout(layout):
    return {'name': layout}


@app.callback(Output('IO-network', 'stylesheet'),
              Output('IO-network', 'elements'),
              [Input('IO-network', 'tapNode')],
              [Input('dropdown-level', 'value')],
              [Input('dropdown-mode', 'value')],
              [State('IO-network', 'elements')])

def generate_stylesheet_expandNode(node,start_level,mode,elements):
    if mode == "selection":
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

        if node['data'].get('selected') == 1:
            try:
                idx = elements.index({"data":node['data']})
            except:
                return default_stylesheet, elements
            elements[idx]['data']['selected'] = 0
        else:
            try:
                idx = elements.index({"data":node['data']})
            except:
                return default_stylesheet, elements
            elements[idx]['data']['selected'] = 1
        
        selected_nodes = [element.get('data').get('id') for element in elements if element.get('data').get('selected') == 1]
        all_edges = [element.get('data') for element in elements if "source" in element['data']]

        stylesheet = [{
            "selector": 'node',
            'style': {
                "label": "data(label)",
                "color": "black",
                "text-opacity": 0.65,
                "font-size": 10,
                'shape': "ellipse"
            }
        }, {
            'selector': 'edge',
            'style': {
                'opacity': 0.3,
                "curve-style": "bezier",
                "width":"data(weight)"
            }
        }, {
            "selector": '[selected > 0]',
            "style": {
                'background-color': '#B10DC9',
                "border-color": "purple",
                "border-width": 2,
                "border-opacity": 1,
                "opacity": 1,

                "label": "data(label)",
                "color": "#B10DC9",
                "text-opacity": 1,
                "font-size": 12,
                'z-index': 9999
            }
        }]
        for edge in all_edges:
            if edge['source'] in selected_nodes:#== node['data']['id']:
                stylesheet.append({
                    "selector": 'node[id = "{}"]'.format(edge['target']),
                    "style": {
                        'background-color': "red",
                        "label": "data(label)",
                        "color": "red",
                        "text-opacity": 1,
                        "font-size": 12,
                        'opacity': 0.9
                    }
                })
                stylesheet.append({
                    "selector": 'edge[id= "{}"]'.format(edge['id']),
                    "style": {
                        "target-arrow-color": "red",
                        "target-arrow-shape": "vee",
                        "line-color": "red",
                        'opacity': 0.9,
                        'z-index': 5000,
                        "width":"data(weight)"
                    }
                })

            if edge['target'] in selected_nodes:#== node['data']['id']:
                stylesheet.append({
                    "selector": 'node[id = "{}"]'.format(edge['source']),
                    "style": {
                        'background-color': "green",
                        "label": "data(label)",
                        "color": "green",
                        "text-opacity": 1,
                        "font-size": 12,
                        'opacity': 0.9,
                        'z-index': 9999
                    }
                })
                stylesheet.append({
                    "selector": 'edge[id= "{}"]'.format(edge['id']),
                    "style": {
                        "target-arrow-color": "green",
                        "target-arrow-shape": "vee",
                        "line-color": "green",
                        'opacity': 1,
                        'z-index': 5000,
                        "width":"data(weight)"
                    }
                })

        return stylesheet,elements

    else:
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

        if selected_level not in ["16","26","58"]:
            return default_stylesheet,elements

        expandedID_deepest = list(nodes.loc[nodes[f"id{selected_level}"] == selected_id,"id"])
        
        io_selected = io_2015.copy()
        node_selected = nodes.copy()

        # delete selected node and associated links from cytoscape element
        new_cytoElement = elements.copy()

        for c in new_cytoElement: # loop to delete
            if selected_id in c['data'].values():
                new_cytoElement.remove(c)

        cyto_node_dict = {}

        last_node_index = 0

        for element in new_cytoElement:
            if 'source' not in element["data"]:
                cyto_node_dict[element["data"]["id"]] = element["data"]["label"]
            else:
                last_node_index += new_cytoElement.index(element)
                last_node_index -= 1
                break

        # get selected node deeper level data
        if selected_level == "16":
            io_selected = io_selected[(io_selected["COLUMN"].isin(expandedID_deepest)) | (io_selected["ROW"].isin(expandedID_deepest))]

            # convert non-selected nodes
            io_selected.loc[~(io_selected["COLUMN"].isin(expandedID_deepest)),"COLUMN"] = io_selected.loc[~(io_selected["COLUMN"].isin(expandedID_deepest)),"COLUMN"].apply(get_all_levels, cyto_node_dict=cyto_node_dict)
            io_selected.loc[~(io_selected["ROW"].isin(expandedID_deepest)),"ROW"] = io_selected.loc[~(io_selected["ROW"].isin(expandedID_deepest)),"ROW"].apply(get_all_levels, cyto_node_dict=cyto_node_dict)

            # convert selected nodes

            io_selected.loc[(io_selected["COLUMN"].isin(expandedID_deepest)),"COLUMN"] = io_selected.loc[(io_selected["COLUMN"].isin(expandedID_deepest)),"COLUMN"].apply(MAP, level="26")
            io_selected.loc[(io_selected["ROW"].isin(expandedID_deepest)),"ROW"] = io_selected.loc[(io_selected["ROW"].isin(expandedID_deepest)),"ROW"].apply(MAP, level="26")

            node_selected = node_selected[node_selected["id"].isin(expandedID_deepest)]
            node_selected = node_selected[["id26","name26"]]
            node_selected.rename(columns={"id26":"id","name26":"name"},inplace=True)
            node_selected.drop_duplicates(inplace=True)

        elif selected_level == "26":
            io_selected = io_selected[(io_selected["COLUMN"].isin(expandedID_deepest)) | (io_selected["ROW"].isin(expandedID_deepest))]

            # convert non-selected nodes
            io_selected.loc[~(io_selected["COLUMN"].isin(expandedID_deepest)),"COLUMN"] = io_selected.loc[~(io_selected["COLUMN"].isin(expandedID_deepest)),"COLUMN"].apply(get_all_levels, cyto_node_dict=cyto_node_dict)
            io_selected.loc[~(io_selected["ROW"].isin(expandedID_deepest)),"ROW"] = io_selected.loc[~(io_selected["ROW"].isin(expandedID_deepest)),"ROW"].apply(get_all_levels, cyto_node_dict=cyto_node_dict)

            # convert selected nodes

            io_selected.loc[(io_selected["COLUMN"].isin(expandedID_deepest)),"COLUMN"] = io_selected.loc[(io_selected["COLUMN"].isin(expandedID_deepest)),"COLUMN"].apply(MAP, level="58")
            io_selected.loc[(io_selected["ROW"].isin(expandedID_deepest)),"ROW"] = io_selected.loc[(io_selected["ROW"].isin(expandedID_deepest)),"ROW"].apply(MAP, level="58")

            node_selected = node_selected[node_selected["id"].isin(expandedID_deepest)]
            node_selected = node_selected[["id58","name58"]]
            node_selected.rename(columns={"id58":"id","name58":"name"},inplace=True)
            node_selected.drop_duplicates(inplace=True)
        
        elif selected_level == "58":
            io_selected = io_selected[(io_selected["COLUMN"].isin(expandedID_deepest)) | (io_selected["ROW"].isin(expandedID_deepest))]

            # convert non-selected nodes
            io_selected.loc[~(io_selected["COLUMN"].isin(expandedID_deepest)),"COLUMN"] = io_selected.loc[~(io_selected["COLUMN"].isin(expandedID_deepest)),"COLUMN"].apply(get_all_levels, cyto_node_dict=cyto_node_dict)
            io_selected.loc[~(io_selected["ROW"].isin(expandedID_deepest)),"ROW"] = io_selected.loc[~(io_selected["ROW"].isin(expandedID_deepest)),"ROW"].apply(get_all_levels, cyto_node_dict=cyto_node_dict)

            # convert selected nodes

            io_selected.loc[(io_selected["COLUMN"].isin(expandedID_deepest)),"COLUMN"] = io_selected.loc[(io_selected["COLUMN"].isin(expandedID_deepest)),"COLUMN"].apply(MAP, level="")
            io_selected.loc[(io_selected["ROW"].isin(expandedID_deepest)),"ROW"] = io_selected.loc[(io_selected["ROW"].isin(expandedID_deepest)),"ROW"].apply(MAP, level="")

            node_selected = node_selected[node_selected["id"].isin(expandedID_deepest)]
            node_selected = node_selected[["id","name"]]
            node_selected.drop_duplicates(inplace=True)

        else:
            return default_stylesheet,elements


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

        new_node_list = []

        for index,row in node_selected.iterrows():
            temp_dict = {}
            temp_dict['data'] = {}
            temp_dict['data']['id'] = row['id']
            temp_dict['data']['label'] = row['name']
            
            if len(row['id']) == 2:
                temp_dict['data']['level'] = "180"
            else:
                temp_dict['data']['level'] = row['id'][-2:]

            new_node_list.append(temp_dict)

        new_cytoElement = new_cytoElement[:last_node_index+1] + new_node_list + new_cytoElement[last_node_index+1:]

        for edge in G_start.edges:
            temp_dict = {}

            temp_dict['data'] = {}
            temp_dict['data']['source'] = edge[0]
            temp_dict['data']['target'] = edge[1]
            temp_dict['data']['weight'] = edge_weight[edge]
            temp_dict['data']['amount'] = amount[edge]

            new_cytoElement.append(temp_dict)
        
        return default_stylesheet,new_cytoElement


@app.callback(Output('node-sanky', 'figure'),
              Output('node-donut-inflow', 'figure'),
              Output('node-donut-outflow', 'figure'),
              Output('all-degree-centrality', 'figure'),
              Output('in-degree-centrality', 'figure'),
              Output('out-degree-centrality', 'figure'),
              [Input('IO-network', 'tapNode')],
              [State('IO-network', 'elements')])

def generate_charts_selectedNode(node,elements):
    if not node: # no node selected
        df_cyto = get_edge_df_from_cyto(elements)
        df_cyto['Buyer_name'] = df_cyto["Buyer"].apply(get_node_name)
        df_cyto['Seller_name'] = df_cyto["Seller"].apply(get_node_name)

        G_cyto = nx.from_pandas_edgelist(df_cyto, 
                            'Buyer_name', 
                            'Seller_name', 
                            ['Amount','Weight'],
                            create_using=nx.DiGraph())

        # defualt chart of all_degree_centrality
        
        degree_centrality = nx.degree_centrality(G_cyto)
        colors = ['lightslategray',] * (len(degree_centrality.keys()))

        DeCen_fig = go.Figure(data=[go.Bar(
            y=list(degree_centrality.keys()),
            x=list(degree_centrality.values()),
            marker_color=colors,
            orientation='h'
        )])


        DeCen_fig.update_layout(title_text='Degree Centralities',yaxis=dict(autorange="reversed"))

        # defualt chart of in_degree_centrality
        
        in_degree_centrality = nx.in_degree_centrality(G_cyto)
        colors = ['lightslategray',] * (len(in_degree_centrality.keys()))

        DeInCen_fig = go.Figure(data=[go.Bar(
            y=list(in_degree_centrality.keys()),
            x=list(in_degree_centrality.values()),
            marker_color=colors,
            orientation='h'
        )])

        DeInCen_fig.update_layout(title_text='In-Degree Centralities',yaxis=dict(autorange="reversed"))

        # defualt chart of out_degree_centrality
        
        out_degree_centrality = nx.out_degree_centrality(G_cyto)
        colors = ['lightslategray',] * (len(out_degree_centrality.keys()))

        DeOutCen_fig = go.Figure(data=[go.Bar(
            y=list(out_degree_centrality.keys()),
            x=list(out_degree_centrality.values()),
            marker_color=colors,
            orientation='h'
        )])

        DeOutCen_fig.update_layout(title_text='Out-Degree Centralities',yaxis=dict(autorange="reversed"))

        return empty_fig, empty_fig, empty_fig, DeCen_fig, DeInCen_fig, DeOutCen_fig
    else:
        selected_id = node["data"]["id"]
        selected_name = node["data"]["label"]

        df_cyto = get_edge_df_from_cyto(elements)
        df_cyto['Buyer_name'] = df_cyto["Buyer"].apply(get_node_name)
        df_cyto['Seller_name'] = df_cyto["Seller"].apply(get_node_name)

        G_cyto = nx.from_pandas_edgelist(df_cyto, 
                            'Buyer_name', 
                            'Seller_name', 
                            ['Amount','Weight'],
                            create_using=nx.DiGraph())

        df_selected = df_cyto[(df_cyto["Buyer"] == selected_id) | (df_cyto["Seller"] == selected_id)]

        if df_selected.shape[0] == 0:
            return empty_fig, empty_fig, empty_fig
        else:
            # sankey
            sankDict = genSankey(df_selected,cat_cols=['Buyer_name','Seller_name'],value_cols='Amount',title=f'Sanky plot of {selected_name}')

            sank_fig = go.Figure(sankDict)

            # donut inflow
            money_inflow_df = df_selected[df_selected["Seller_name"] == selected_name].groupby(["Buyer_name","Seller_name"]).sum().reset_index()

            money_inflow_df.sort_values("Amount",ascending=False,inplace=True)
            money_inflow_df.iloc[5:,0] = "Other"

            sum_inflow = money_inflow_df["Amount"].sum()
            sum_inflow = "{:,}".format(sum_inflow)

            money_inflow_df.rename(columns={"Buyer_name":"Buyer Name",},inplace=True)

            donut_inflow_fig = px.pie(money_inflow_df, values="Amount", names="Buyer Name", hole=.65)
            donut_inflow_fig.update_layout(title_text=f"Money Inflow of {selected_name}",
                            annotations=[dict(text=f'฿{sum_inflow}', x=0.5, y=0.5, font_size=20, showarrow=False)])

            # donut outflow
            money_outflow_df = df_selected[df_selected["Buyer_name"] == selected_name].groupby(["Buyer_name","Seller_name"]).sum().reset_index()

            money_outflow_df.sort_values("Amount",ascending=False,inplace=True)
            money_outflow_df.iloc[5:,1] = "Other"

            sum_outflow = money_outflow_df["Amount"].sum()
            sum_outflow = "{:,}".format(sum_outflow)

            money_outflow_df.rename(columns={"Seller_name":"Seller Name",},inplace=True)

            donut_outflow_fig = px.pie(money_outflow_df, values="Amount", names="Seller Name", hole=.65)
            donut_outflow_fig.update_layout(title_text=f"Money Outflow of {selected_name}",
                            annotations=[dict(text=f'฿{sum_outflow}', x=0.5, y=0.5, font_size=20, showarrow=False)])

            
            # centralities

            degree_centrality = nx.degree_centrality(G_cyto)
            colors = ['lightslategray',] * (len(degree_centrality.keys()))
            colors[list(degree_centrality.keys()).index(selected_name)] = 'crimson'

            DeCen_fig = go.Figure(data=[go.Bar(
                y=list(degree_centrality.keys()),
                x=list(degree_centrality.values()),
                marker_color=colors,
                orientation='h'
            )])

            DeCen_fig.update_layout(title_text='Degree Centralities',yaxis=dict(autorange="reversed"))

            in_degree_centrality = nx.in_degree_centrality(G_cyto)
            colors = ['lightslategray',] * (len(in_degree_centrality.keys()))
            colors[list(in_degree_centrality.keys()).index(selected_name)] = 'crimson'

            DeInCen_fig = go.Figure(data=[go.Bar(
                y=list(in_degree_centrality.keys()),
                x=list(in_degree_centrality.values()),
                marker_color=colors,
                orientation='h'
            )])

            DeInCen_fig.update_layout(title_text='In-Degree Centralities',yaxis=dict(autorange="reversed"))

            out_degree_centrality = nx.out_degree_centrality(G_cyto)
            colors = ['lightslategray',] * (len(out_degree_centrality.keys()))
            colors[list(out_degree_centrality.keys()).index(selected_name)] = 'crimson'

            DeOutCen_fig = go.Figure(data=[go.Bar(
                y=list(out_degree_centrality.keys()),
                x=list(out_degree_centrality.values()),
                marker_color=colors,
                orientation='h'
            )])

            DeOutCen_fig.update_layout(title_text='Out-Degree Centralities',yaxis=dict(autorange="reversed"))


            return sank_fig, donut_inflow_fig, donut_outflow_fig, DeCen_fig, DeInCen_fig, DeOutCen_fig

if __name__ == '__main__':
    app.run_server(debug=True)