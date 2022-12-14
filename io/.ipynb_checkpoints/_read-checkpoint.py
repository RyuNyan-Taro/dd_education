import pandas as pd
import numpy as np
from pandas import DataFrame
from typing import Tuple
from sklearn.preprocessing import LabelEncoder

from ._config import *


def read_data(data='train') -> pd.DataFrame:
	"""
	read data then set index and reindex columns
	
	Parameters
	----------
	data : {‘train’, ‘test, subformat’}, default='train'
		data type to read
	
	Returns
	-------
	df_sorted: pd.DataFrame
		DataFrame which setted index and reindexed columns
	
	"""
	
	data_list = ['train', 'test', 'subformat']
	if data not in data_list:
		raise ValueError(f'You should select {data_list}')
	if data == 'train':
		csv_path = TRAIN_PATH
	elif data == 'test':
		csv_path = TEST_PATH
	elif data == 'subformat':
		df = pd.read_csv(SUBFORMAT_PATH, index_col=['Unnamed: 0'])
		return df
	df = pd.read_csv(csv_path, index_col=['Unnamed: 0']
					).sort_index(ascending=True)
	if data == 'train':
		col_sort = sorted(df.columns[9:].tolist()) + sorted(df.columns[:9].tolist())
	elif data == 'test':
		col_sort = sorted(df.columns.tolist())
	
	return df.reindex(columns=col_sort)


def devide_xy(df: pd.DataFrame) -> Tuple[DataFrame, np.array, list]:
	"""
	Return x and y 
	
	Parameters
	----------
	df : DataFrame
		Dataframe to use
		
	Returns
	-------
	x_df : DataFrame
		x DataFrame
	y_array : np.array
		y 1-D array
	feature_cols : list
		feature cols list
	"""
	
	x_df = df.iloc[:, :16]
	feature_cols = list(np.where(x_df.dtypes != float)[0])
	x_df = df.iloc[:, list(set(range(len(x_df.columns))) - set(feature_cols))
				  ].merge(df.iloc[:, feature_cols].fillna('NO LABEL'), left_index=True, right_index=True)
	feature_cols = list(np.where(x_df.dtypes != float)[0])
	le = LabelEncoder()
	y_array = le.fit_transform(df.iloc[:, 16])
	
	return x_df, y_array, feature_cols
	
	