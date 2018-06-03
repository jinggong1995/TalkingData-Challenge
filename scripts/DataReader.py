import pandas as pd  
import gc
import time
import multiprocessing as mp
import numpy as np 
from sklearn import preprocessing
from featureDict import FeatureDictionary
from getFeature import GetFeature




def ReadData(example, label, train=False):



	# convert array into dataframe
	if train:
		df = pd.DataFrame(example, columns=['ip','app','device','os', 'channel', 'day','hour','minute'])
		df['is_attributed'] = label
	else:
		df = pd.DataFrame(example, columns=['ip','app','device','os', 'channel', 'day','hour','minute', 'click_id'])


	# covert click_time

	year = np.array(2017-1970, dtype='<M8[Y]')
	month = np.array(11-1, dtype='<m8[M]')
	day = np.array(example[:, 5]-1, dtype='<m8[D]')
	hour = np.array(example[:, 6], dtype='<m8[h]')
	minute = np.array(example[:, 7], dtype='<m8[m]')


	df['click_time'] = year+month+day+hour+minute


	df.drop(['day','hour','minute'], axis=1, inplace=True)


	'''---- df as dataframe for lgbm input  --- '''
	print 'getting features', time.clock()

	df = GetFeature(df).df # write dataframe to csv file

	print 'perprocessing data', time.clock()


	'''---- df as dataframe for DeepFM input  --- '''

	# fill missing values 
	na_col = df.columns[df.isna().any()].tolist()
	if 'is_attributed' in df.columns:
		for col in na_col:
			df[col] = df.groupby('is_attributed').transform(lambda x: x.fillna(x.mean()))

	else:
		df[na_col] = df[na_col].fillna(value = df[na_col].mean())



	# scale numeric data between 0 and 1 
	numeric_cols =  df.select_dtypes(include=[np.float]).columns.tolist()
	scaler = preprocessing.MinMaxScaler()
	df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

	#  get feature index and feature value 
	if train:
		ignore_cols='is_attributed'
	else:
		ignore_cols='click_id'

	n_ip = 400000
	n_app = 800
	n_device = 5000
	n_os = 900
	n_channel = 600
	n_count = 300
	n_day = 30
	n_hour = 24

	n_size = {'ip': n_ip, 'app': n_app, 'device': n_device,
		'day': n_day, 'hour': n_hour,
		'os': n_os, 'channel': n_channel, 'ucount_channel_ip': n_count,
		'mcount_app_ip_device_os': n_count, 'ucount_hour_ip_day': n_count,
		'ucount_app_ip': n_count, 'u_count_app_ip': n_count, 'ucount_os_ip_app': n_count, 
		'ucount_device_ip': n_count, 'ucount_channel_app': n_count, 'mcount_os_ip': n_count,
		'ucount_os_app': n_count, 'count_t_ip': n_count, 'count_ip_app': n_count,
		'count_ip_app_os': n_count}


	print 'indexing', time.clock()


	fd = FeatureDictionary(df, numeric_cols, ignore_cols, n_size)
	Xi, Xv, y = fd.get_feat_index(df, train=True)  # input as a batch of data to DeepFM


	return Xi, Xv, y





if __name__ == '__main__':
	dtypes = {
			'ip'            : 'uint32',
			'app'           : 'uint16',
			'device'        : 'uint16',
			'os'            : 'uint16',
			'channel'       : 'uint16',
			'is_attributed' : 'uint8',
			'click_id'      : 'uint32',
			}

	train=True
	frm = 1
	to = 4000000
	if train:
		df = pd.read_csv('train.csv', skiprows=range(1, frm), 
			nrows = to-frm, dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'is_attributed'])
		df['day']=1
		df['hour']=2
		df['minute']=1

	print df.head()


	label = df['is_attributed'].values
	df.drop(['is_attributed'], axis=1, inplace=True)
	example = df.values

	print 'start', time.clock()

	Xi, Xv, y = ReadData(example, label, train=True)


	print 'end', time.clock()












