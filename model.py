from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np

def model(X_train, X_test, y_train, y_test):
    model = XGBRegressor(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(y_pred)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    importance = model.feature_importances_
    feat_imp = pd.Series(importance, index=X_train.columns)
    feat_imp = feat_imp.sort_values(ascending=False)
    
    return mse, rmse, mae, feat_imp

def model_predict(X, y):
    X_train = X.iloc[:-1]
    X_test = X.iloc[-1:]
    y_train = y.iloc[:-1]

    model = XGBRegressor(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)[0]

    return y_pred
