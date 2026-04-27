import lightgbm as lgb
import numpy as np


def train_lightgbm(X_train, y_train, params=None, num_boost_round=100):
    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
        }
    dtrain = lgb.Dataset(X_train, label=y_train)
    booster = lgb.train(params, dtrain, num_boost_round=num_boost_round)
    return booster


def predict_lightgbm(model, X):
    return model.predict(X)
