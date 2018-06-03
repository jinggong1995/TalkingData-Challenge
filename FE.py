import pandas as pd  
import gc
import time
import numpy as np 
from sklearn import preprocessing
import dask.dataframe as dd
import dask.multiprocessing
import psutil




def add_count(df, group_col, agg_name, agg_type='uint32'):
	gp = df[group_col].groupby(group_col).size().rename(agg_name).to_frame().reset_index()
	gc.collect()
	df = df.merge(gp, on=group_col, how='left')
	del gp
	gc.collect()
	df[agg_name] = df[agg_name].astype(agg_type)
	print ('group', agg_name, 'by', group_col, time.clock())
	print('memory used', str(int(psutil.virtual_memory().used / (1024.0 ** 3))) +'GB')
	gc.collect()
	return df


def add_unique_count(df, group_col, counted, agg_name, agg_type='int'):
	#df[group_col, counted].groupby(df[group_col]).counted.nunique()
	gp = df[group_col+[counted]].groupby(group_col)[counted].nunique().reset_index().rename(columns={counted:agg_name})
	gc.collect()
	df = df.merge(gp, on=group_col, how='left')
	del gp
	gc.collect()
	df[agg_name] = df[agg_name].astype(agg_type)
	gc.collect()
	print ('group', agg_name, 'by', group_col, time.clock())
	print('memory used', str(int(psutil.virtual_memory().used / (1024.0 ** 3))) +'GB')
	return df


def add_cumulative_count(df, group_col, counted, agg_name, agg_type='uint32'):
	gp = df[group_col+[counted]].groupby(group_col)[counted].cumcount()
	gc.collect()
	df[agg_name] = gp.values
	del gp
	gc.collect()
	df[agg_name] = df[agg_name].astype(agg_type)
	gc.collect()
	print ('group', agg_name, 'by', group_col, time.clock())
	print('memory used', str(int(psutil.virtual_memory().used / (1024.0 ** 3))) +'GB')
	return df



def add_mean(df, group_col, counted, agg_name, agg_type='float32'):
	gp = df[group_col+[counted]].groupby(group_col)[counted].mean().reset_index().rename(columns={counted:agg_name})
	gc.collect()
	df = df.merge(gp, on=group_col, how='left')
	del gp
	gc.collect()
	df[agg_name] = df[agg_name].astype(agg_type)
	print ('group', agg_name, 'by', group_col, time.clock())
	print('memory used', str(int(psutil.virtual_memory().used / (1024.0 ** 3))) +'GB')
	gc.collect()
	return df



def add_var(df, group_col, counted, agg_name, agg_type='float32'):
	gp = df[group_col+[counted]].groupby(group_col)[counted].var().reset_index().rename(columns={counted:agg_name})
	gc.collect()
	df = df.merge(gp, on=group_col, how='left')
	del gp
	gc.collect()
	df[agg_name] = df[agg_name].astype(agg_type) 
	print ('group', agg_name, 'by', group_col, time.clock())
	print('memory used', str(int(psutil.virtual_memory().used / (1024.0 ** 3))) +'GB')
	gc.collect()
	return df



def minMax(x):
	return pd.Series(index=['min','max'],data=[x.min(),x.max()])


if __name__ == '__main__':

	'''

	dtypes = {
			'ip'            : 'uint32',
			'app'           : 'uint16',
			'device'        : 'uint16',
			'os'            : 'uint16',
			'channel'       : 'uint16',
			'is_attributed' : 'uint8',
			'click_id'      : 'uint32',
			}


	# for full train
	nrows=184903891-1
	'''


	print('loading train data...', time.clock())
	train_df = dd.read_csv("train_sample.csv", parse_dates=['click_time'])
	gc.collect()
	train_df = train_df.drop('attributed_time', axis=1)
	gc.collect()

	print('memory used', str(int(psutil.virtual_memory().used / (1024.0 ** 3))) +'GB')
	
	#train_df = train_df.compute()
	gc.collect()


	train_df = train_df.compute(get=dask.multiprocessing.get)
	gc.collect()
	train_df['is_attributed'] = train_df['is_attributed'].astype(int)
	gc.collect()
	
	''''
	
	print('loading test data...', time.clock())
	test_df = dd.read_csv("test.csv", parse_dates=['click_time'])
	#test_df = test_df.assign(day = test_df.click_time.dt.day.astype(int))
	#test_df = test_df.assign(hour = test_df.click_time.dt.hour.astype(int))
	test_df = test_df.compute(get=dask.multiprocessing.get)
	print('memory used', str(int(psutil.virtual_memory().used / (1024.0 ** 3))) +'GB')
	gc.collect()

	#df = train_temp.compute(get=dask.multiprocessing.get)


	len_train = len(train_df)
	train_df=train_df.append(test_df)
	del test_df
	print (len_train)
	
	gc.collect()
	'''
	

	print (train_df.head())

	

	print('getting features...', time.clock())


	# Extract new features
	train_df['day'] = train_df['click_time'].dt.day.astype(int)
	gc.collect()
	train_df['hour'] = train_df['click_time'].dt.hour.astype(int)
	print(train_df.head())
	gc.collect()
		#self.df = self.df.assign(day = self.df.click_time.dt.day.astype(int))
		#self.df = self.df.assign(hour = self.df.click_time.dt.hour.astype(int))

		# Group by aggregation(mean/variance/count/unique_count/cumulative_count)

	print ('aggregation', time.clock())

	train_df = add_unique_count(train_df, ['ip'], 'channel', 'ucount_channel_ip', 'int')
	gc.collect()
	print (train_df.head())


	train_df = add_cumulative_count(train_df, ['ip', 'device', 'os'], 'app', 'mcount_app_ip_device_os', 'int')
	print (train_df.head())
	gc.collect()
	

	train_df = add_unique_count(train_df, ['ip', 'day'], 'hour', 'ucount_hour_ip_day', 'int')
	print (train_df.head())
	gc.collect()

	train_df = add_unique_count(train_df, ['ip'], 'app', 'ucount_app_ip', 'int')
	print (train_df.head())
	gc.collect()


	train_df = add_unique_count(train_df, ['ip', 'app'], 'os', 'ucount_os_ip_app', 'int')
	print (train_df.head())
	gc.collect()

	train_df = add_unique_count(train_df, ['ip'], 'device', 'ucount_device_ip', 'int')
	print (train_df.head())
	gc.collect()
	

	train_df = add_unique_count(train_df, ['app'], 'channel', 'ucount_channel_app')
	print (train_df.head())
	gc.collect()

	train_df = add_cumulative_count(train_df, ['ip'], 'os', 'mcount_os_ip')
	print (train_df.head())
	gc.collect()


	train_df = add_unique_count(train_df, ['ip', 'device', 'os'], 'app', 'ucount_os_app')
	print (train_df.head())
	gc.collect()

	train_df = add_count(train_df, ['ip', 'day', 'hour'], 'count_t_ip')
	print (train_df.head())
	gc.collect()


	train_df = add_count(train_df, ['ip', 'app'], 'count_ip_app')
	print (train_df.head())
	gc.collect()

	train_df = add_count(train_df, ['ip', 'app', 'os'], 'count_ip_app_os', 'int')
	print (train_df.head())
	gc.collect()


	train_df = add_var(train_df, ['ip', 'day', 'channel'], 'hour', 'var_hour_ip_tchan')
	print (train_df.head())
	gc.collect()

	train_df = add_var(train_df, ['ip', 'app', 'os'], 'hour', 'var_hour_ip_app_os')
	print (train_df.head())
	gc.collect()


	train_df = add_var(train_df, ['ip', 'app', 'channel'], 'day', 'var_day_ip_app_channel')
	print (train_df.head())
	gc.collect()


	train_df = add_mean(train_df, ['ip', 'app', 'channel'], 'hour', 'mean_hour_ip_app_channel', 'float')
	print (train_df.head())
	gc.collect()



	# Creat new features by calcuating the time untill next click
	print ('create new feature', time.clock())

	predictors=[]
	new_feature = 'nextClick'
	D = 2**26


	# to numeric 
	#self.df['category']
	m = ((train_df['ip'].astype(str) + "_" + train_df['app'].astype(str) + 
		"_" + train_df['device'].astype(str) + "_" + train_df['os'].astype(str)).apply(hash) % D).values
	click_buffer = np.full(D, 3000000000, dtype=np.uint32)
	gc.collect()
	print('memory used', str(int(psutil.virtual_memory().used / (1024.0 ** 3))) +'GB')

	#self.df['epochtime']
	n = (train_df['click_time'].astype(np.int64) // 10 ** 9).values
	gc.collect()
	next_clicks = []
	for category, t in zip(reversed(m), reversed(n)):
		next_clicks.append(click_buffer[category]-t)
		click_buffer[category] = t
	del click_buffer
	gc.collect()
	del m
	gc.collect()
	del n
	gc.collect()
	QQ = list(reversed(next_clicks))
	gc.collect()
	print('memory used', str(int(psutil.virtual_memory().used / (1024.0 ** 3))) +'GB')
	print (train_df.head())

	# drop columns
	'''
	self.df.drop(['epochtime'], axis = 1, inplace=True)
	gc.collect()
	self.df.drop(['category'], axis = 1, inplace=True)
	gc.collect()
	self.df.drop(['click_time'], axis = 1, inplace=True)
	gc.collect()
	'''


	# add new features to dataframe
	train_df[new_feature] = pd.Series(QQ).astype('float32')
	del QQ
	gc.collect()

	train_df[new_feature+'_shift'] = train_df[new_feature].shift(+1).values
	gc.collect()

	train_df['nextClick'] = train_df['nextClick'].astype(float)
	train_df['mean_hour_ip_app_channel'] = train_df['mean_hour_ip_app_channel'].astype(float)



	gc.collect()
	print('memory used', str(int(psutil.virtual_memory().used / (1024.0 ** 3))) +'GB')
	mm = train_df.apply(minMax)
	print mm

	print train_df.dtypes
	'''


	print("FE done", time.clock())
	print(train_df.head())



	train_df.to_csv('new_train.csv.gz',compression='gzip', chunksize=10000)
	

	train_df[:len_train].to_csv('new_train.csv')
	train_df[len_train:].to_csv('new_test.csv')


	'''


