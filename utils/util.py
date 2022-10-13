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