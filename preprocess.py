import pandas as pd  
import gc
import time
import numpy as np 
from sklearn import preprocessing
import dask.dataframe as dd
import dask.multiprocessing
import dask.threaded


def read_csv(csv_file):
	cols = ['app', 'channel', 'device', 'ip',
		 'os', 'day', 'hour', 'ucount_channel_ip', 'mcount_app_ip_device_os', 
		 'ucount_hour_ip_day', 'ucount_app_ip', 
		 'ucount_os_ip_app', 'ucount_device_ip', 'ucount_channel_app', 'mcount_os_ip',
		 'ucount_os_app', 'count_t_ip', 'count_ip_app',
		  'count_ip_app_os', 'var_hour_ip_tchan', 'var_hour_ip_app_os', 
		  'var_day_ip_app_channel', 'mean_hour_ip_app_channel', 'nextClick', 'nextClick_shift']

	df = dd.read_csv(csv_file, blocksize=None)
	#df = df.repartition(npartitions= 1000)
	gc.collect()
	#print('1', time.clock())
	#df = df.compute()
	#df = df.map_partitions(df.compute())
	df = df.compute(get=dask.threaded.get)
	#df = df.compute(get=dask.multiprocessing.get)
	print('2', time.clock())
	gc.collect()
	label = df.is_attributed.values
	X = df[cols].values
	del df
	gc.collect()
	return X, label




def get_index(X):

	#numeric_index = [ 19, 20, 21, 23, 24]

	#size = [770, 510, 4230, 365000, 960, 12, 24, 170,282430, 25, 280, 150, 555, 50,  1421260,
	#110, 44260, 220745, 55160, 1, 1, 1, 24, 1, 1] 

	numeric_index = range(7, 22).append([23, 24])
	size = [770, 510, 4230, 365000, 960, 12, 24, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 24, 1, 1]

	 #10, 155, 282430, 69, 99, 49, 1421260, 58, 1, 10547, 2186, 1, 1, 1, 24, 1, 1]

	# fill in missing values with mean of the column
	col_mean = np.nanmean(X, axis=0)
	inds = np.where(np.isnan(X))
	X[inds] = np.take(col_mean, inds[1])
	del col_mean
	del inds


	# scale numeric column to range(0, 1)
	scaler = preprocessing.MinMaxScaler()
	mm = scaler.fit_transform(X[:, numeric_index])
	X[:, numeric_index] = mm
	del mm
	gc.collect()

	# indexing
	Xi = X.copy()
	Xv = X.copy()
	k = 0
	for col in range(X.shape[1]):
		if col in numeric_index:
			Xi[:, col] = k
			k+= size[col]
		else:
			Xi[:, col] =  Xi[:, col] + k
			Xv[:, col] = 1
			k += size[col]
	feat_size = k
	#print('feature_size', k)
	Xi = Xi.astype(int)
	return Xi, Xv



if __name__ == '__main__':


	size = [770, 510, 4230, 365000, 960, 12, 24, 170,282430, 25, 280, 150, 555, 50,  1421260,
	110, 44260, 220745, 55160, 1, 1, 1, 24, 1, 1] 
	print sum(size)
	print len(size)


	print (0.10 * 2000  * 12 * 60 * 60/ (86400  * 30 )) 

	print(time.clock())


	X, label = read_csv('new_train.csv')
	print (X.shape)
	print(X[2, :])
	print (label)
	gc.collect()
	
	'''

	print(time.clock())
	Xi, Xv = get_index(X)
	print Xi[3]
	print Xv[3]
	gc.collect()
	print Xi.shape
	print Xv.shape
	'''

	'''


	cols = ['app', 'channel', 'device', 'ip',
		 'os', 'day', 'hour', 'ucount_channel_ip', 
		 'ucount_hour_ip_day', 'ucount_app_ip', 
		 'ucount_os_ip_app', 'ucount_device_ip', 'ucount_channel_app', 
		 'ucount_os_app', 'count_t_ip', 'count_ip_app',
		  'count_ip_app_os', 'var_hour_ip_tchan', 'var_hour_ip_app_os', 
		  'var_day_ip_app_channel', 'mean_hour_ip_app_channel', 'nextClick', 'nextClick_shift']

	numeric_cols = ['var_hour_ip_tchan','var_hour_ip_app_os','var_day_ip_app_channel','nextClick','nextClick_shift']


	n_size = {'ip': 365000, 'app': 770, 'device': 4230,'day': 12, 'hour': 24,
			'os': 960, 'channel': 510, 'ucount_channel_ip': 151,'ucount_hour_ip_day': 10,
			'ucount_app_ip': 155, 'ucount_os_ip_app': 69, 'ucount_device_ip': 99, 'ucount_channel_app': 49, 
			'ucount_os_app': 58, 'count_t_ip': 1, 'count_ip_app': 10547,
			'count_ip_app_os': 2186, 'mean_hour_ip_app_channel ': 24, 
			'var_hour_ip_tchan': 1, 'var_hour_ip_app_os': 1, 'var_day_ip_app_channel': 1,
			'mean_hour_ip_app_channel': 24, 'nextClick': 1, 'nextClick_shift':1}


	# load compressed csv file


	train_df = dd.read_csv('new_train.csv.gz', compression='gzip', blocksize=None).compute()

	label = train_df.is_attributed.values
	train_df = train_df[cols]
	X = train_df.values
	total_col = train_df.columns
	del train_df
	gc.collect()
	

	# get column index
	numeric_index = [total_col.get_loc(c) for c in numeric_cols]
	print numeric_index
	col_index = [total_col.get_loc(c) for c in cols]
	size = [n_size[col] for col in total_col]
	print size

	# get examples 

	#preprocessing and indexing
	Xi, Xv = preprocess(X, numeric_index, size)
	print Xi[:2]
	print Xv[:2]
	gc.collect()
	print Xi.shape
	print Xv.shape
	'''





	'''
		Col = ['app', 'channel', 'click_id', 'click_time', 'device', 'ip',
		 'is_attributed', 'os', 'day', 'hour', 'ucount_channel_ip', 
		 'mcount_app_ip_device_os', 'ucount_hour_ip_day', 'ucount_app_ip', 
		 'ucount_os_ip_app', 'ucount_device_ip', 'ucount_channel_app', 
		 'mcount_os_ip', 'ucount_os_app', 'count_t_ip', 'count_ip_app',
		  'count_ip_app_os', 'var_hour_ip_tchan', 'var_hour_ip_app_os', 
		  'var_day_ip_app_channel', 'mean_hour_ip_app_channel', 'nextClick', 
		  'nextClick_shift']

	
		n_app = 770
		n_channel = 510
		n_device = 4230
		n_ip = 365000
		n_os = 960
		n_day = 12
		n_hour = 24
		ucount_channel_ip = 151
		mcount_app_ip_device_os = 0
		ucount_hour_ip_day = 10
		ucount_app_ip = 155
		ucount_os_ip_app = 69
		ucount_device_ip = 99
		ucount_channel_app = 49
		mcount_os_ip = 0
		ucount_os_app = 58
		count_t_ip = 1
		count_ip_app = 10547
		count_ip_app_os = 2186
		mean_hour_ip_app_channel = 24



	'''










