import warnings

import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

import methods

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

import constants



def preprocess_df(df_i: pd.DataFrame):
    df = df_i.copy()
    temp_ghi_columns = ['Site-1 Temp', 'Site-2 Temp',
                        'Site-3 Temp', 'Site-4 Temp', 'Site-5 Temp', 'Site-1 GHI', 'Site-2 GHI',
                        'Site-3 GHI', 'Site-4 GHI', 'Site-5 GHI']

    TOTAL_LAG = 48
    LAG_INTERVAL = 3
    for lag in range(3, TOTAL_LAG + 1, LAG_INTERVAL):
        for col in temp_ghi_columns:
            df[col + f'_lag_{lag}'] = df[col].shift(lag)
    df = df.drop(columns=['Day'])
    df = df.dropna()
    return df


def train_model(df: pd.DataFrame):
    X_train = df.drop(columns=['Load', 'Year'])
    y_train = df['Load']
    xgb_params = {'subsample': 0.2, 'n_estimators': 350, 'max_depth': 7, 'learning_rate': 0.017,
                  'colsample_bytree': 0.25}
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train)
    return xgb_model




if __name__ == "__main__":
    df = methods.load_df()
    df_preprocessed = preprocess_df(df)

    methods.print_partition("PK_Method")

    # Generate Test 1 Predictions
    train_df = df[df['Year'] == 2]
    test_df = df[df['Year'] == 1]
    preprocess_train_df = df_preprocessed[df_preprocessed['Year'] == 2]
    preprocess_test_df = df_preprocessed[df_preprocessed['Year'] == 1]

    model = train_model(preprocess_train_df)
    methods.generate_excel(train_df, preprocess_train_df, model, constants.PK_TRAIN_2, constants.PK_COL)
    methods.generate_excel(test_df, preprocess_test_df, model, constants.PK_TEST_1, constants.PK_COL)

    # Generate Test 2 Predictions
    train_df = df[df['Year'] == 1]
    test_df = df[df['Year'] == 2]
    preprocess_train_df = df_preprocessed[df_preprocessed['Year'] == 1]
    preprocess_test_df = df_preprocessed[df_preprocessed['Year'] == 2]

    model = train_model(preprocess_train_df)
    methods.generate_excel(train_df, preprocess_train_df, model, constants.PK_TRAIN_1, constants.PK_COL)
    methods.generate_excel(test_df, preprocess_test_df, model, constants.PK_TEST_2, constants.PK_COL)

    # Generate Test 3 Predictions
    train_df = df[df['Year'] < 3]
    test_df = df[df['Year'] == 3]
    preprocess_train_df = df_preprocessed[df_preprocessed['Year'] < 3]
    preprocess_test_df = df_preprocessed[df_preprocessed['Year'] == 3]

    model = train_model(preprocess_train_df)
    methods.generate_excel(train_df, preprocess_train_df, model, constants.PK_TRAIN_1_2, constants.PK_COL)
    methods.generate_excel(test_df, preprocess_test_df, model, constants.PK_TEST_3, constants.PK_COL)
