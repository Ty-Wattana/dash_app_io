import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

nodes = pd.read_excel('io_data/node.xlsx', dtype={'id':'object', 'id58':'object','id26':'object','id16':'object'})

irr_node = ['190','201','202','203','204','209','210','301','302','303','304',
            '305','306','309','310','401','402','403','404','409','501','502', '503',
            '509','600','700']

nodes = nodes[~nodes['id'].isin(irr_node)]
nodes['id58'] = nodes['id58'] + "_58"
nodes['id26'] = nodes['id26'] + "_26"
nodes['id16'] = nodes['id16'] + "_16"


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


def get_all_levels(id_deepest,cyto_node_dict):
  all_level_df = nodes[nodes["id"] == id_deepest][["id","id58","id26","id16"]]
  all_level_df = all_level_df.iloc[0,:]
  all_level_list = all_level_df.to_list()

  diff = np.setdiff1d(all_level_list,list(cyto_node_dict.keys()))
  res = np.setdiff1d(all_level_list,diff)
  res = res[0]
  
  return res

def get_node_name(id):
  if len(id) > 3:
    level = id[-2:]
    return nodes.loc[nodes[f"id{level}"] == id,f"name{level}"].values[0]
  else:
    return nodes.loc[nodes["id"] == id,"name"].values[0]


def get_edge_df_from_cyto(cyto):
  Buyer = []
  Seller = []
  Amount = []
  Weight = []

  for element in cyto:
      if "source" in element["data"]:
        Buyer.append(element["data"]["source"])
        Seller.append(element["data"]["target"])
        Amount.append(element["data"]["amount"])
        Weight.append(element["data"]["weight"])

  df_cyto = pd.DataFrame({
      "Buyer":Buyer,
      "Seller":Seller,
      "Amount":Amount,
      "Weight":Weight,
  })

  return df_cyto


def genSankey(df,cat_cols=[],value_cols='',title='Sankey Diagram'):
    # maximum of 6 value cols -> 6 colors
    colorPalette = ['#4B8BBE','#306998','#FFE873','#FFD43B','#646464']
    labelList = []
    colorNumList = []
    for catCol in cat_cols:
        labelListTemp =  list(set(df[catCol].values))
        colorNumList.append(len(labelListTemp))
        labelList = labelList + labelListTemp
        
    # remove duplicates from labelList
    labelList = list(dict.fromkeys(labelList))
    
    # define colors based on number of levels
    colorList = []
    for idx, colorNum in enumerate(colorNumList):
        colorList = colorList + [colorPalette[idx]]*colorNum
        
    # transform df into a source-target pair
    for i in range(len(cat_cols)-1):
        if i==0:
            sourceTargetDf = df[[cat_cols[i],cat_cols[i+1],value_cols]]
            sourceTargetDf.columns = ['source','target','count']
        else:
            tempDf = df[[cat_cols[i],cat_cols[i+1],value_cols]]
            tempDf.columns = ['source','target','count']
            sourceTargetDf = pd.concat([sourceTargetDf,tempDf])
        sourceTargetDf = sourceTargetDf.groupby(['source','target']).agg({'count':'sum'}).reset_index()
        
    # add index for source-target pair
    sourceTargetDf['sourceID'] = sourceTargetDf['source'].apply(lambda x: labelList.index(x))
    sourceTargetDf['targetID'] = sourceTargetDf['target'].apply(lambda x: labelList.index(x))
    
    # creating the sankey diagram
    data = dict(
        type='sankey',
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(
            color = "black",
            width = 0.5
          ),
          label = labelList,
          color = colorList
        ),
        link = dict(
          source = sourceTargetDf['sourceID'],
          target = sourceTargetDf['targetID'],
          value = sourceTargetDf['count']
        )
      )
    
    layout =  dict(
        title = title,
        font = dict(
          size = 10
        )
    )
       
    fig = dict(data=[data], layout=layout)
    return fig