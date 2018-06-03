import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
import matplotlib.pyplot as plt
import os
import dask.dataframe as dd
import dask.multiprocessing


def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', 
	metrics='auc', feval=None, early_stopping_rounds=20, num_boost_round=3000, 
	verbose_eval=10, categorical_features=None):
	
	lgb_params = {
	'boosting_type': 'gbdt',
	'objective': objective,
	'metric': metrics,
	'learning_rate': 0.2,
	'num_leaves': 31,
	'max_depth': -1,
	'min_child_samples': 20,
	'max_bin': 255,
	'subsample': 0.6,
	'subsample_freq': 0,
	'colsample_bytree': 0.3,
	'min_child_weight': 5,
	'subsample_for_bin': 200000,
	'min_split_gain': 0,
	'reg_alpha': 0,
	'reg_lamda': 0,
	'nthread': 8,
	'verbose': 0
	}

	lgb_params.update(params)

	print("preparing validation datasets")


	xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
		feature_name=predictors, categorical_feature = categorical_features)

	xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
		feature_name=predictors, categorical_feature=categorical_features)

	del dtrain
	del dvalid
	gc.collect()

	evals_results = {}

	bst1 = lgb.train(lgb_params, xgtrain, valid_sets=xgvalid, valid_names=['valid'],
		evals_result=evals_results, num_boost_round=num_boost_round,
		early_stopping_rounds=early_stopping_rounds, verbose_eval=10, feval=feval)

	print("\nModel Report")
	print("bst1.best_iteration: ", bst1.best_iteration)
	print(metrics+":", evals_results['valid'][metrics][bst1.best_iteration-1])

	return (bst1,bst1.best_iteration)


def DO(train_df, val_df, test_df):

	predictors = ['app', 'channel', 'device', 'ip',
		 'os', 'day', 'hour', 'ucount_channel_ip', 'mcount_app_ip_device_os',
		 'ucount_hour_ip_day', 'ucount_app_ip', 
		 'ucount_os_ip_app', 'ucount_device_ip', 'ucount_channel_app', 'mcount_os_ip',
		 'ucount_os_app', 'count_t_ip', 'count_ip_app',
		  'count_ip_app_os', 'var_hour_ip_tchan', 'var_hour_ip_app_os', 
		  'var_day_ip_app_channel', 'mean_hour_ip_app_channel', 'nextClick', 'nextClick_shift']

	#categorical = ['app', 'device', 'os', 'channel', 'hour']

	print("\ntrain size: ", len(train_df))
	print("\nvalid size: ", len(val_df))
	print("\ntest size : ", len(test_df))

	sub = pd.DataFrame()
	sub['click_id'] = test_df['click_id'].astype('int')

	gc.collect()

	print("Training...")
	start_time = time.time()

	params = {
		'learning_rate': 0.20,
		#'is_unbalance': 'true', # replaced with scale_pos_weight argument
		'num_leaves': 7,  # 2^max_depth - 1
		'max_depth': 3,  # -1 means no limit
		'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
		'max_bin': 100,  # Number of bucketed bin for feature values
		'subsample': 0.7,  # Subsample ratio of the training instance.
		'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
		'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
		'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
		'scale_pos_weight':200 # because training data is extremely unbalanced 
		}

	(best, best_iteration) = lgb_modelfit_nocv(params,train_df,val_df, predictors,target,
		objective='binary', metrics='auc', early_stopping_rounds=30, verbose_eval=True,
		num_boost_round=1000,categorical_features=categorical)

	print('[{}]: model training time'.format(time.time() - start_time))
	del train_df
	del val_df
	gc.collect()


	print('Plot feature importance...')

	ax = lgb.plot_importance(bst, max_num_features=300)
	#plt.savefig('test%d.png'%(fileno), dpi=600,bbox_inches="tight")
	plt.show()


	print('Predicting...')

	sub['is_attributed'] = bst.predict(test_df[predictors],num_iteration=best_iteration)

	sub.to_csv('sub_it%d.csv'%(0),index=False,float_format='%.9f')

	print("done...")
	return sub


if __name__ == '__main__':

	nchunk=40000000

	begin_time = time.time()

	print('loading train data')

	train_df = dd.read_csv('new_train.csv', parse_dates=['click_time'], sample=nchunk)
	train_df = train_df.compute(get=dask.multiprocessing.get)
	gc.collect()

	print('[{}]: train data loading time'.format(time.time() - begin_time))

	print('loading test data')

	test_df = dd.read_csv('new_test.csv', parse_dates=['click_time'])
	test_df = test_df.compute(get=dask.multiprocessing.get)

	print('[{}]: test data loading time'.format(time.time() - begin_time))

	val_size=2500000
	val_df = train_df[(nchunk-val_size):]
	train_df = train_df[: (nchunk-val_size)]



	sub = DO(train_df, val_df, test_df)











