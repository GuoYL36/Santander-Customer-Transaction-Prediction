# -----------XGBoost------------
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import time
# import tensorflow as tf
# from sklearn.model_selection import StratifiedKFold
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn import metrics
# from xgboost.sklearn import XGBClassifier
# from sklearn.model_selection import train_test_split
# import xgboost as xgb
#
# df_train = pd.read_csv('./train.csv')
# df_test = pd.read_csv('./test.csv')
# df_submission = pd.read_csv('./sample_submission.csv')
#
# # Y=df_train['target']
# # X=df_train.drop(['ID_code','target'], axis=1)
# #
# # trains, targets = X.values, Y.values
# # train_x, valid_x, train_y, valid_y = train_test_split(trains, targets, test_size=0.1, random_state=40)
# target = 'target'
# predictors = [x for x in df_train.columns if x not in ['ID_code','target']]
#
# # 定义一个函数来产生XGBoost模型及其效果
#
# def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
#     if useTrainCV:
#         xgb_param = alg.get_xgb_params()
#         xgbtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
#         cvresult = xgb.cv(xgb_param, xgbtrain, num_boost_round=alg.get_params()['n_estimators'],
#                           nfold=cv_folds, metrics='auc', early_stopping_rounds=early_stopping_rounds)
#         alg.set_params(n_estimators=cvresult.shape[0])
#
#     # Fit the algorithm on the data
#     alg.fit(dtrain[predictors], dtrain[target], eval_metric='auc')
#
#     # Predict training set
#     dtrain_predictions = alg.predict(dtrain[predictors])
#     dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]
#
#     # Print model report
#     print("\nModel Report")
#     print("Accuracy: %.4g" % (metrics.accuracy_score(dtrain[target].values, dtrain_predictions)))
#     print("AUC Score (Train): %f" % (metrics.roc_auc_score(dtrain[target], dtrain_predprob)))
#
#     feat_imp = pd.Series(alg.booster().feature_importances_).sort_values(ascending=False)
#     feat_imp.plot(kind='bar', title='Feature Importances')
#     plt.ylabel("Feature Importance Score")
#     plt.show()
#
# # def metrics_score(gbm):
# #     accu_train = gbm.score(train_x, train_y)
# #     auc_score = metrics.roc_auc_score(train_y, gbm.predict_proba(train_x)[:,1])
# #     print('train accuracy:',accu_train, 'train auc:',auc_score)
# #     accu_valid = gbm.score(valid_x, valid_y)
# #     auc_score = metrics.roc_auc_score(valid_y, gbm.predict_proba(valid_x)[:,1])
# #     print('valid accuracy:', accu_valid, 'valid auc:',auc_score)
#
# xgb1 = XGBClassifier(learning_rate=0.1,n_estimators=200,max_depth=5,
#                     min_child_weight=1,gamma=0,subsample=0.8,colsample_bytree=0.8,
#                     objective='binary:logistic',nthread=4,scale_pos_weight=1,seed=27)
# modelfit(xgb1, df_train,predictors)


#-----------lightgbm0----------------
#
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import StratifiedKFold
#
# from sklearn import metrics
#
# import lightgbm as lgb
#
#
# df_train = pd.read_csv('./train.csv')
# df_test = pd.read_csv('./test.csv')
# df_submission = pd.read_csv('./sample_submission.csv')
#
# features = df_train.columns.values[2:202]
# correlations = df_train[features].corr().abs().unstack().sort_values(kind='quicksort').reset_index()
# correlations = correlations[correlations['level_0'] != correlations['level_1']]
#
# features = df_train.columns.values[2:202]
# unique_max_train = []
# unique_max_test = []
# for feature in features:
#     values = df_train[feature].value_counts()
#     unique_max_train.append([feature,values.max(), values.idxmax()])
#     values = df_test[feature].value_counts()
#     unique_max_test.append([feature,values.max(),values.idxmax()])
#
# idx = features = df_train.columns.values[2:202]
# for df in [df_test, df_train]:
#     df['sum'] = df[idx].sum(axis=1)
#     df['min'] = df[idx].min(axis=1)
#     df['max'] = df[idx].max(axis=1)
#     df['mean'] = df[idx].mean(axis=1)
#     df['std'] = df[idx].std(axis=1)
#     df['skew'] = df[idx].skew(axis=1)
#     df['kurt'] = df[idx].kurtosis(axis=1)
#     df['med'] = df[idx].median(axis=1)
#
# features = [c for c in df_train.columns if c not in ['ID_code', 'target']]
# for feature in features:
#     df_train['r2_'+feature] = np.round(df_train[feature], 2)
#     df_test['r2_'+feature] = np.round(df_test[feature], 2)
#     df_train['r1_'+feature] = np.round(df_train[feature], 1)
#     df_test['r1_'+feature] = np.round(df_test[feature], 1)
#
#
# features = [c for c in df_train.columns if c not in ['ID_code', 'target']]
# target = df_train['target']
#
#
# param = {
#     'bagging_freq': 5,
#     'bagging_fraction': 0.4,
#     'boost_from_average':'false',
#     'boost': 'gbdt',
#     'feature_fraction': 0.05,
#     'learning_rate': 0.01,
#     'max_depth': -1,
#     'metric':'auc',
#     'min_data_in_leaf': 80,
#     'min_sum_hessian_in_leaf': 10.0,
#     'num_leaves': 13,
#     'num_threads': 8,
#     'tree_learner': 'serial',
#     'objective': 'binary',
#     'verbosity': 1
# }
#
# folds = StratifiedKFold(n_splits=10, shuffle=False, random_state=44000)
# oof = np.zeros(len(df_train))
# predictions = np.zeros(len(df_test))
# feature_importance_df = pd.DataFrame()
#
# for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train.values, target.values)):
#     print("Fold {}".format(fold_))
#     trn_data = lgb.Dataset(df_train.iloc[trn_idx][features], label=target.iloc[trn_idx])
#     val_data = lgb.Dataset(df_train.iloc[val_idx][features], label=target.iloc[val_idx])
#
#     num_round = 1000000
#     clf = lgb.train(param, trn_data, num_round,
#                     valid_sets=[trn_data, val_data],
#                     verbose_eval=1000, early_stopping_rounds=3000)
#     oof[val_idx] = clf.predict(df_train.iloc[val_idx][features],
#                                num_iteration=clf.best_iteration)
#     fold_importance_df = pd.DataFrame()
#     fold_importance_df["Feature"] = features
#     fold_importance_df["importance"] = clf.feature_importance()
#     fold_importance_df['fold'] = fold_ + 1
#     feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
#
#     predictions += clf.predict(df_test[features], num_iteration=clf.best_iteration) / folds.n_splits
#
# print("CV score: {:<8.5f}".format(metrics.roc_auc_score(target, oof)))
#
# clf.save_model('model.txt')
#
#
# sub_df = pd.DataFrame({"ID_code":df_test["ID_code"].values})
# sub_df['target'] = predictions
# sub_df.to_csv("submission.csv",index=False)





#-----------lightgbm1----------------
'''
    在lightgbm0代码的基础上，增加了augment操作
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

from sklearn import metrics

import lightgbm as lgb

df_train = pd.read_csv('./train.csv')
df_test = pd.read_csv('./test.csv')
df_submission = pd.read_csv('./sample_submission.csv')

features = df_train.columns.values[2:202]


features = [c for c in df_train.columns if c not in ['ID_code', 'target']]
target = df_train['target']

def augment(x,y,t=2):
    xs,xn = [],[]
    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xs.append(x1)        #对label为1的特征进行两遍shuffle，相当于增加了2倍的label为1的数据

    for i in range(t//2):
        mask = y==0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xn.append(x1)        #对label为0的特征进行一遍shuffle，相当于增加了1倍的label为0的数据

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x,xs,xn])  # 扩增数据，label为1的数据变为原来label为1的3倍，label为0的数据变为原来label为0的2倍
    y = np.concatenate([y,ys,yn])
    return x,y

param = {
    'bagging_freq': 5,
    'bagging_fraction': 0.4,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.05,
    'learning_rate': 0.01,
    'max_depth': -1,
    'metric':'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary',
    'verbosity': 1
}

folds = StratifiedKFold(n_splits=10, shuffle=False, random_state=44000)
# oof = df_train[['ID_code', 'target']].copy()
# oof['predict'] = np.zeros(len(df_test))
predictions = df_test[["ID_code"]].copy()
feature_importance_df = pd.DataFrame()
val_aucs = []
for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train.values, target.values)):

    X_train, y_train = df_train.iloc[trn_idx][features], target.iloc[trn_idx]
    X_valid, y_valid = df_train.iloc[val_idx][features], target.iloc[val_idx]
    N = 5
    p_valid, yp = 0, 0
    for i in range(N):
        print("Fold idx:{}".format(i + 1))
        X_t, y_t = augment(X_train.values, y_train.values)
        X_t = pd.DataFrame(X_t)
        X_t = X_t.add_prefix('var_')

        trn_data = lgb.Dataset(X_t, label=y_t)
        val_data = lgb.Dataset(X_valid, label=y_valid)
        evals_result = {}
        lgb_clf = lgb.train(param,
                            trn_data,
                            100000,
                            valid_sets=[trn_data, val_data],
                            early_stopping_rounds=3000,
                            verbose_eval=1000,
                            evals_result=evals_result
                            )
        p_valid += lgb_clf.predict(X_valid)
        yp += lgb_clf.predict(df_test[features].values)

    # oof.loc[val_idx, 'predict'] = p_valid /N
    val_score = metrics.roc_auc_score(y_valid, p_valid)
    val_aucs.append(val_score)
    # oof[val_idx] = p_valid / N
    predictions['fold{}'.format(fold_ + 1)] = yp / N

    # submission
predictions['target'] = np.mean(predictions[[col for col in predictions.columns if col not in ['ID_code', 'target']]].values, axis=1)
# predictions.to_csv('lgb_all_predictions.csv', index=None)
sub_df = pd.DataFrame({"ID_code": df_test["ID_code"].values})
sub_df["target"] = predictions['target']
sub_df.to_csv("add_lgb_submission.csv", index=False)
# oof.to_csv('lgb_oof.csv', index=False)