import warnings

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

import constants


def load_df():
    train_df = pd.read_excel(constants.ESD_TRAINING)
    test_df = pd.read_excel(constants.ESD_TESTING)
    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    df = df.fillna(0)
    return df


def preprocess_df(df_i: pd.DataFrame):
    df = df_i.copy()
    df['Day'] = np.round(3*df['Day']/32,0)
    return df


def train_model(df: pd.DataFrame):
    X_train = df.drop(columns=['Load'])
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


def generate_excel(df: pd.DataFrame, preprocess_df: pd.DataFrame, model, file_name: str):
    X = preprocess_df.drop(columns=['Load'])
    y = model.predict(X)
    X[constants.NG_COL] = y
    df = df.join(X[constants.NG_COL], how='left')
    df_no_na = df.dropna()
    r2 = r2_score(df_no_na['Load'], df_no_na[constants.NG_COL])
    rmse = root_mean_squared_error(df_no_na['Load'], df_no_na[constants.NG_COL])
    mae = mean_absolute_error(df_no_na['Load'], df_no_na[constants.NG_COL])
    print(f"{file_name:<30} {str(df_no_na.shape):<10}\t->\t R^2: {r2:.2f} \t RMSE: {rmse:.2f} \t MAE: {mae:.2f}")
    df.to_excel(file_name, index=False)


def print_partition(title=""):
    print("\n" + "=" * 50)
    if title:
        print(f"{title.center(50)}")
        print("=" * 50)


if __name__ == "__main__":
    df = load_df()
    df_preprocessed = preprocess_df(df)

    print_partition("NG_Method")

    # Generate Test 1 Predictions
    train_df = df[df['Year'] == 2]
    test_df = df[df['Year'] == 1]
    preprocess_train_df = df_preprocessed[df_preprocessed['Year'] == 2]
    preprocess_test_df = df_preprocessed[df_preprocessed['Year'] == 1]

    model = train_model(preprocess_train_df)
    generate_excel(train_df, preprocess_train_df, model, constants.NG_TRAIN_2)
    generate_excel(test_df, preprocess_test_df, model, constants.NG_TEST_1)

    # Generate Test 2 Predictions
    train_df = df[df['Year'] == 1]
    test_df = df[df['Year'] == 2]
    preprocess_train_df = df_preprocessed[df_preprocessed['Year'] == 1]
    preprocess_test_df = df_preprocessed[df_preprocessed['Year'] == 2]

    model = train_model(preprocess_train_df)
    generate_excel(train_df, preprocess_train_df, model, constants.NG_TRAIN_1)
    generate_excel(test_df, preprocess_test_df, model, constants.NG_TEST_2)

    # Generate Test 3 Predictions
    train_df = df[df['Year'] < 3]
    test_df = df[df['Year'] == 3]
    preprocess_train_df = df_preprocessed[df_preprocessed['Year'] < 3]
    preprocess_test_df = df_preprocessed[df_preprocessed['Year'] == 3]

    model = train_model(preprocess_train_df)
    generate_excel(train_df, preprocess_train_df, model, constants.NG_TRAIN_1_2)
    generate_excel(test_df, preprocess_test_df, model, constants.NG_TEST_3)
