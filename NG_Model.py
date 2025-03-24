import warnings

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

import methods

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

import constants



def preprocess_df(df_i: pd.DataFrame):
    df = df_i.copy()
    df['Day'] = np.round(3*df['Day']/32,0)
    return df


def train_model(df: pd.DataFrame):
    X_train = df.drop(columns=['Load', 'Year'])
    y_train = df['Load']

    xgb_model = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=1.0,
        colsample_bytree=0.6,
        objective='reg:squarederror',
        random_state=42
    )

    xgb_model.fit(X_train, y_train)
    return xgb_model





if __name__ == "__main__":
    df = methods.load_df()
    df_preprocessed = preprocess_df(df)

    methods.print_partition("NG_Method")

    # Generate Test 1 Predictions
    train_df = df[df['Year'] == 2]
    test_df = df[df['Year'] == 1]
    preprocess_train_df = df_preprocessed[df_preprocessed['Year'] == 2]
    preprocess_test_df = df_preprocessed[df_preprocessed['Year'] == 1]

    model = train_model(preprocess_train_df)
    methods.generate_excel(train_df, preprocess_train_df, model, constants.NG_TRAIN_2, constants.NG_COL)
    methods.generate_excel(test_df, preprocess_test_df, model, constants.NG_TEST_1, constants.NG_COL)

    # Generate Test 2 Predictions
    train_df = df[df['Year'] == 1]
    test_df = df[df['Year'] == 2]
    preprocess_train_df = df_preprocessed[df_preprocessed['Year'] == 1]
    preprocess_test_df = df_preprocessed[df_preprocessed['Year'] == 2]

    model = train_model(preprocess_train_df)
    methods.generate_excel(train_df, preprocess_train_df, model, constants.NG_TRAIN_1, constants.NG_COL)
    methods.generate_excel(test_df, preprocess_test_df, model, constants.NG_TEST_2, constants.NG_COL)

    # Generate Test 3 Predictions
    train_df = df[df['Year'] < 3]
    test_df = df[df['Year'] == 3]
    preprocess_train_df = df_preprocessed[df_preprocessed['Year'] < 3]
    preprocess_test_df = df_preprocessed[df_preprocessed['Year'] == 3]

    model = train_model(preprocess_train_df)
    methods.generate_excel(train_df, preprocess_train_df, model, constants.NG_TRAIN_1_2, constants.NG_COL)
    methods.generate_excel(test_df, preprocess_test_df, model, constants.NG_TEST_3, constants.NG_COL)
