import pandas as pd  
import gc
import time
import numpy as np 
from sklearn import preprocessing
import dask.dataframe as dd
import dask.multiprocessing
import psutil


begin_time = time.time()


print('loading train data...')
train_df = pd.read_csv("new_train.csv", parse_dates=['click_time'])
gc.collect()
print('memory used', str(int(psutil.virtual_memory().used / (1024.0 ** 3))) +'GB')
print (train_df.head())
train_time = time.time()
print('[{}]: train data loading time'.format(train_time - begin_time))



print('loading test data...')
test_df = pd.read_csv('new_test.csv', parse_dates=['click_time'])
gc.collect()
print('memory used', str(int(psutil.virtual_memory().used / (1024.0 ** 3))) +'GB')
print (test_df.head())
print('[{}]: train data loading time'.format(time.time() - train_time))



len_train = len(train_df)
train_df=train_df.append(test_df)
del test_df
gc.collect()
print (len_train)

# drop day
train_df = train_df.drop('day', axis=1)
gc.collect()




print('new feature for group(ip, app, device, channel, os)')
time_1 = time.time()
# next click for group(ip, app, device, channel, os)
new_feature = 'nextClick_1'
D = 2**26
n = (train_df['click_time'].astype(np.int64) // 10 ** 9).values


m = ((train_df['ip'].astype(str) + "_" + train_df['app'].astype(str) + 
	"_" + train_df['device'].astype(str) + "_" + train_df['channel'].astype(str) + "_" + 
	train_df['os'].astype(str)).apply(hash) % D).values
click_buffer = np.full(D, 3000000000, dtype=np.uint32)
gc.collect()
print('memory used', str(int(psutil.virtual_memory().used / (1024.0 ** 3))) +'GB')


gc.collect()
next_clicks = []
for category, t in zip(reversed(m), reversed(n)):
	next_clicks.append(click_buffer[category]-t)
	click_buffer[category] = t
del click_buffer
gc.collect()
del m
gc.collect()
QQ = list(reversed(next_clicks))
gc.collect()
print('memory used', str(int(psutil.virtual_memory().used / (1024.0 ** 3))) +'GB')
train_df[new_feature] = pd.Series(QQ).astype('float32')
del QQ
gc.collect()
train_df[new_feature+'_shift'] = train_df[new_feature].shift(+1).values
gc.collect()
print('[{}]: new feature 1 done'.format(time.time() - time_1))






time_2 = time.time()
# group(ip, os, device)
new_feature = 'nextClick_2'

m = ((train_df['ip'].astype(str) + "_" + train_df['device'].astype(str) + "_" + 
	train_df['os'].astype(str)).apply(hash) % D).values
click_buffer = np.full(D, 3000000000, dtype=np.uint32)
gc.collect()
print('memory used', str(int(psutil.virtual_memory().used / (1024.0 ** 3))) +'GB')

next_clicks = []
for category, t in zip(reversed(m), reversed(n)):
	next_clicks.append(click_buffer[category]-t)
	click_buffer[category] = t
del click_buffer
gc.collect()
del m
gc.collect()
QQ = list(reversed(next_clicks))
gc.collect()
print('memory used', str(int(psutil.virtual_memory().used / (1024.0 ** 3))) +'GB')
train_df[new_feature] = pd.Series(QQ).astype('float32')
del QQ
gc.collect()
train_df[new_feature+'_shift'] = train_df[new_feature].shift(+1).values
gc.collect()
print('[{}]: new feature 2 done'.format(time.time() - time_2))



time_3 = time.time()
# group(app, device, channel)
new_feature = 'nextClick_3'

m = ((train_df['app'].astype(str) + "_" + train_df['device'].astype(str) + "_" + 
	train_df['channel'].astype(str)).apply(hash) % D).values
click_buffer = np.full(D, 3000000000, dtype=np.uint32)
gc.collect()
print('memory used', str(int(psutil.virtual_memory().used / (1024.0 ** 3))) +'GB')

next_clicks = []
for category, t in zip(reversed(m), reversed(n)):
	next_clicks.append(click_buffer[category]-t)
	click_buffer[category] = t
del click_buffer
gc.collect()
del m
gc.collect()
QQ = list(reversed(next_clicks))
gc.collect()
print('memory used', str(int(psutil.virtual_memory().used / (1024.0 ** 3))) +'GB')
train_df[new_feature] = pd.Series(QQ).astype('float32')
del QQ
gc.collect()
train_df[new_feature+'_shift'] = train_df[new_feature].shift(+1).values
gc.collect()
print('[{}]: new feature 3 done'.format(time.time() - time_3))

print(train_df.head())



# write to csv
train_df[:len_train].to_csv('new_train_2.csv')
train_df[len_train:].to_csv('new_test_2.csv')






