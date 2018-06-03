# -*- coding: utf-8 -*-
import pandas as pd   
import time
import numpy as np 
import gc
import psutil



class GetFeature():

	def __init__(self, df):
		self.df = df
		del df
		gc.collect()
		self.generate_features()


	def add_count(self, group_col, agg_name, agg_type='uint32'):
		gp = self.df[group_col].groupby(group_col).size().rename(agg_name).to_frame().reset_index()
		self.df = self.df.merge(gp, on=group_col, how='left')
		del gp
		gc.collect()
		self.df[agg_name] = self.df[agg_name].astype(agg_type)
		print ('group', agg_name, 'by', group_col, time.clock())
		print('memory used', str(int(psutil.virtual_memory().used / (1024.0 ** 3))) +'GB')


	def add_unique_count(self, group_col, counted, agg_name, agg_type='int'):
		#df[group_col, counted].groupby(df[group_col]).counted.nunique()
		gp = self.df[group_col+[counted]].groupby(group_col)[counted].nunique().reset_index().rename(columns={counted:agg_name})
		self.df = self.df.merge(gp, on=group_col, how='left')
		del gp
		gc.collect()
		self.df[agg_name] = self.df[agg_name].astype(agg_type)
		gc.collect()
		print ('group', agg_name, 'by', group_col, time.clock())
		print('memory used', str(int(psutil.virtual_memory().used / (1024.0 ** 3))) +'GB')


	def add_cumulative_count(self, group_col, counted, agg_name, agg_type='uint32'):
		gp = self.df[group_col+[counted]].groupby(group_col)[counted].cumcount()
		self.df[agg_name] = gp.values
		del gp
		gc.collect()
		self.df[agg_name] = self.df[agg_name].astype(agg_type)
		print ('group', agg_name, 'by', group_col, time.clock())
		print('memory used', str(int(psutil.virtual_memory().used / (1024.0 ** 3))) +'GB')



	def add_mean(self, group_col, counted, agg_name, agg_type='float32'):
		gp = self.df[group_col+[counted]].groupby(group_col)[counted].mean().reset_index().rename(columns={counted:agg_name})
		self.df = self.df.merge(gp, on=group_col, how='left')
		del gp
		gc.collect()
		self.df[agg_name] = self.df[agg_name].astype(agg_type)
		print ('group', agg_name, 'by', group_col, time.clock())
		print('memory used', str(int(psutil.virtual_memory().used / (1024.0 ** 3))) +'GB')



	def add_var(self, group_col, counted, agg_name, agg_type='float32'):
		gp = self.df[group_col+[counted]].groupby(group_col)[counted].var().reset_index().rename(columns={counted:agg_name})
		self.df = self.df.merge(gp, on=group_col, how='left')
		del gp
		gc.collect()
		self.df[agg_name] = self.df[agg_name].astype(agg_type) 
		print ('group', agg_name, 'by', group_col, time.clock())
		print('memory used', str(int(psutil.virtual_memory().used / (1024.0 ** 3))) +'GB')



	def generate_features(self):


		# Extract new features
		self.df['day'] = self.df.click_time.dt.day.astype(int)
		self.df['hour'] = self.df.click_time.dt.hour.astype(int)
		gc.collect()
		#self.df = self.df.assign(day = self.df.click_time.dt.day.astype(int))
		#self.df = self.df.assign(hour = self.df.click_time.dt.hour.astype(int))

		# Group by aggregation(mean/variance/count/unique_count/cumulative_count)

		print ('aggregation', time.clock())

		self.add_unique_count(['ip'], 'channel', 'ucount_channel_ip', 'int')
		gc.collect()
		print (self.df.head())


		self.add_cumulative_count(['ip', 'device', 'os'], 'app', 'mcount_app_ip_device_os', 'int')
		print (self.df.head())
		gc.collect()
		

		self.add_unique_count(['ip', 'day'], 'hour', 'ucount_hour_ip_day', 'int')
		print (self.df.head())
		gc.collect()

		self.add_unique_count(['ip'], 'app', 'ucount_app_ip', 'int')
		print (self.df.head())
		gc.collect()
	

		self.add_unique_count(['ip', 'app'], 'os', 'ucount_os_ip_app', 'int')
		print (self.df.head())
		gc.collect()

		self.add_unique_count(['ip'], 'device', 'ucount_device_ip', 'int')
		print (self.df.head())
		gc.collect()
		

		self.add_unique_count(['app'], 'channel', 'ucount_channel_app')
		print (self.df.head())
		gc.collect()

		self.add_cumulative_count(['ip'], 'os', 'mcount_os_ip')
		print (self.df.head())
		gc.collect()


		self.add_unique_count(['ip', 'device', 'os'], 'app', 'ucount_os_app')
		print (self.df.head())
		gc.collect()

		self.add_count(['ip', 'day', 'hour'], 'count_t_ip')
		print (self.df.head())
		gc.collect()
	

		self.add_count(['ip', 'app'], 'count_ip_app')
		print (self.df.head())
		gc.collect()

		self.add_count(['ip', 'app', 'os'], 'count_ip_app_os', 'int')
		print (self.df.head())
		gc.collect()


		self.add_var(['ip', 'day', 'channel'], 'hour', 'var_hour_ip_tchan')
		print (self.df.head())
		gc.collect()

		self.add_var(['ip', 'app', 'os'], 'hour', 'var_hour_ip_app_os')
		print (self.df.head())
		gc.collect()


		self.add_var(['ip', 'app', 'channel'], 'day', 'var_day_ip_app_channel')
		print (self.df.head())
		gc.collect()


		self.add_mean(['ip', 'app', 'channel'], 'hour', 'mean_hour_ip_app_channel', 'float')
		print (self.df.head())
		gc.collect()


		'''

		GROUP_BY_NEXT_CLICKS = [
			
			# V1
			#{'groupby': ['ip']},
			#{'groupby': ['ip', 'app']},
			#{'groupby': ['ip', 'channel']},
			#{'groupby': ['ip', 'os']},
			
			# V3
			#{'groupby': ['ip', 'app', 'device', 'os', 'channel']},
			{'groupby': ['ip', 'os', 'device']},
			{'groupby': ['ip', 'os', 'device', 'app']}
		]

		# Calculate the time to next click for each group
		for spec in GROUP_BY_NEXT_CLICKS:
			
			# Name of new feature
			new_feature = '{}_nextClick'.format('_'.join(spec['groupby']))    
			
			# Unique list of features to select
			all_features = spec['groupby'] + ['click_time']
			
			# Run calculation
			print('group')
			self.df[new_feature] = self.df[all_features].groupby(spec['groupby']).click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds


		gc.collect()
		'''

	
		# Creat new features by calcuating the time untill next click
		print ('create new feature', time.clock())

		predictors=[]
		new_feature = 'nextClick'
		D = 2**26


		# to numeric 
		#self.df['category']
		m = ((self.df['ip'].astype(str) + "_" + self.df['app'].astype(str) + 
			"_" + self.df['device'].astype(str) + "_" + self.df['os'].astype(str)).apply(hash) % D).values
		click_buffer = np.full(D, 3000000000, dtype=np.uint32)
		gc.collect()
		print('memory used', str(int(psutil.virtual_memory().used / (1024.0 ** 3))) +'GB')
		print (self.df.head())

		#self.df['epochtime']
		n = (self.df['click_time'].astype(np.int64) // 10 ** 9).values
		gc.collect()
		next_clicks = []
		for category, t in zip(reversed(m), reversed(n)):
			next_clicks.append(click_buffer[category]-t)
			click_buffer[category] = t
		del click_buffer
		del m
		del n
		QQ = list(reversed(next_clicks))
		gc.collect()
		print('memory used', str(int(psutil.virtual_memory().used / (1024.0 ** 3))) +'GB')
		print (self.df.head())

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
		self.df[new_feature] = pd.Series(QQ).astype('float32')
		del QQ
		gc.collect()

		self.df[new_feature+'_shift'] = self.df[new_feature].shift(+1).values
		gc.collect()

		self.df['nextClick'] = self.df['nextClick'].astype(float)
		self.df['mean_hour_ip_app_channel'] = self.df['mean_hour_ip_app_channel'].astype(float)

		gc.collect()
	







