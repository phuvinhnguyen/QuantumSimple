import xgboost as xgb
import numpy as np
import pandas as pd


class XGBoost_regresion:
    def __init__(self,
                 objective='reg:squarederror',
                 eval_metric='rmse',
                 n_estimators=100,
                 max_depth=5,
                 learning_rate=0.1
                 ):
        self.xgb = xgb.XGBRegressor(
            objective=objective,
            eval_metric=eval_metric,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate
            )

    def predict(self, input):
        return self.xgb.predict(input)
    
    def fit(self, input, output):
        self.xgb.fit(input, output)
