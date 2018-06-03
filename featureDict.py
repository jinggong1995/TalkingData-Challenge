import pandas as pd 
import gc
import time

class FeatureDictionary():



	def __init__(self, df, numeric_cols, ignore_cols, n_size):
		self.numeric_cols = numeric_cols
		self.ignore_cols = ignore_cols
		self.n_size = n_size



	def get_feat_index(self, df, train=False):
		dfi = df.copy()
	
		if train:
			y = dfi['is_attributed'].values
			dfi.drop(['is_attributed'], axis=1, inplace=True)
		else:
			ids = dfi['click_id'].values
			dfi.drop(['click_id'], axis=1, inplace=True)
		
		# dfi for feaure index
		# dfv for feature value(binary(0/1) or float)
		dfv = dfi.copy()
		k = 0
		for col in dfi.columns:
			if col in self.ignore_cols:
				dfi.drop(col, axis=1, inplace=True)
				dfv.drop(col, axis=1, inplace=True)
				continue
			if col in self.numeric_cols:
				dfi[col] = k
				k += 1
			else:
				dfi.loc[df[col] < self.n_size[col], col] = df.loc[df[col] < self.n_size[col], col] + k
				dfi.loc[df[col] >= self.n_size[col], col] = self.n_size[col] - 1 + k
				k += self.n_size[col]
				dfv[col] = 1
		self.feat_size = k
		Xi = dfi.values
		Xv = dfv.values
		del dfi
		del dfv
		del df
		if train:
			return Xi, Xv, y
		else:
			return Xi, Xv, ids










