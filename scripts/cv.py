import re
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from multiprocessing import Pool
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, cross_val_predict
from lightgbm.sklearn import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, f1_score, log_loss
import gc


def train(x_train, y_train, x_valid, y_valid):


    usecols = x_train.columns.values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=871)

    all_params = {'min_child_weight': [25],
                  'subsample': [0.7],
                  'subsample_freq': [1],
                  'seed': [114],
                  'colsample_bytree': [0.6],
                  'learning_rate': [0.1],
                  'max_depth': [-1],
                  'min_split_gain': [0.001],
                  'reg_alpha': [0.0001],
                  'max_bin': [2047],
                  'num_leaves': [127],
                  'objective': ['binary'],
                  'metric': [['binary_logloss', 'auc']],
                  'scale_pos_weight': [1],
                  'verbose': [-1],
                  }

    use_score = 0
    min_score = (100, 100, 100)

    for params in tqdm(list(ParameterGrid(all_params))):
        cnt = -1
        list_score = []
        list_score2 = []
        list_best_iter = []
        all_pred = np.zeros(y_train.shape[0])
        if 1:
            cnt += 1
            trn_x = x_train
            val_x = x_valid
            trn_y = y_train
            val_y = y_valid
            train_data = lgb.Dataset(trn_x.values.astype(np.float32), label=trn_y,
                                     categorical_feature=CAT_FEAT, feature_name=x_train.columns.values.tolist())
            test_data = lgb.Dataset(val_x.values.astype(np.float32), label=val_y,
                                    categorical_feature=CAT_FEAT, feature_name=x_train.columns.values.tolist())
            del trn_x
            gc.collect()
            clf = lgb.train(params,
                            train_data,
                            10000,  # params['n_estimators'],
                            early_stopping_rounds=30,
                            valid_sets=[test_data],
                            # feval=cst_metric_xgb,
                            # callbacks=[callback],
                            verbose_eval=10
                            )
            pred = clf.predict(val_x)

            #all_pred[test] = pred

            _score2 = log_loss(val_y, pred)
            _score = - roc_auc_score(val_y, pred)

            logger.info('   _score: %s' % _score)
            logger.info('   _score2: %s' % _score2)

            list_score.append(_score)
            list_score2.append(_score2)

            if clf.best_iteration != 0:
                list_best_iter.append(clf.best_iteration)
            else:
                list_best_iter.append(params['n_estimators'])
            gc.collect()

        score = (np.mean(list_score), np.min(list_score), np.max(list_score))
        score2 = (np.mean(list_score2), np.min(list_score2), np.max(list_score2))

        if min_score[use_score] > score[use_score]:
            min_score = score
            min_params = params


    imp = pd.DataFrame(clf.feature_importance(), columns=['imp'])
    imp['col'] = usecols
    n_features = imp.shape[0]
    imp = imp.sort_values('imp', ascending=False)
    imp.to_csv(DIR + 'feature_importances_0.csv')

    del val_x
    del trn_y
    del val_y
    del train_data
    del test_data
    gc.collect()

    trees = np.mean(list_best_iter)

    x_train = pd.concat([x_train, x_valid], axis=0, ignore_index=True)
    y_train = np.r_[y_train, y_valid]
    del x_valid
    del y_valid
    gc.collect()

    train_data = lgb.Dataset(x_train.values.astype(np.float32), label=y_train,
                             categorical_feature=CAT_FEAT, feature_name=x_train.columns.values.tolist())
    del x_train
    gc.collect()

    clf = lgb.train(min_params,
                    train_data,
                    int(trees * 1.1),
                    valid_sets=[train_data],
                    verbose_eval=10
                    )

    #del x_train
    gc.collect()
    return min_params
