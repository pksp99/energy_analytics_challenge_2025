import warnings

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

import constants
from collections import defaultdict


def load_df():
    train_df = pd.read_excel(constants.ESD_TRAINING)
    test_df = pd.read_excel(constants.ESD_TESTING)
    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    df = df.fillna(0)
    return df

def add_feature_engineering(df_i: pd.DataFrame):
    df = df_i.copy()
    df['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 365)
    df['Day_cos'] = np.cos(2 * np.pi * df['Day'] / 365)
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    return df
def preprocess_df(df_i: pd.DataFrame):
    df = add_feature_engineering(df_i)
    seq_len = 24
    loads = df.pop('Load')
    years = df.pop('Year')
    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df)
    custom_index = []
    data_dict = defaultdict(list)

    # Create Flat Window
    for i in range(len(df) - seq_len):
        flattened = df.values[i: i + seq_len].flatten()
        custom_index.append(loads.index[i + seq_len])
        for i_name, feature in enumerate(flattened):
            data_dict[i_name].append(feature)

    n_df = pd.DataFrame(data_dict, index=custom_index)
    n_df = n_df.join(loads, how='inner').join(years, how='inner')
    return n_df


def train_model(df: pd.DataFrame):
    X_train = df.drop(columns=['Load', 'Year'])
    y_train = df['Load']
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10)
    rf_model.fit(X_train, y_train)
    return rf_model


def generate_excel(df: pd.DataFrame, preprocess_df: pd.DataFrame, model, file_name: str):
    X = preprocess_df.drop(columns=['Load', 'Year'])
    y = model.predict(X)
    X[constants.SA_COL] = y
    df = df.join(X[constants.SA_COL], how='left')
    df_no_na = df.dropna()
    r2 = r2_score(df_no_na['Load'], df_no_na[constants.SA_COL])
    rmse = root_mean_squared_error(df_no_na['Load'], df_no_na[constants.SA_COL])
    mae = mean_absolute_error(df_no_na['Load'], df_no_na[constants.SA_COL])
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

    print_partition("SA_Method")

    # Generate Test 1 Predictions
    train_df = df[df['Year'] == 2]
    test_df = df[df['Year'] == 1]
    preprocess_train_df = df_preprocessed[df_preprocessed['Year'] == 2]
    preprocess_test_df = df_preprocessed[df_preprocessed['Year'] == 1]

    model = train_model(preprocess_train_df)
    generate_excel(train_df, preprocess_train_df, model, constants.SA_TRAIN_2)
    generate_excel(test_df, preprocess_test_df, model, constants.SA_TEST_1)

    # Generate Test 2 Predictions
    train_df = df[df['Year'] == 1]
    test_df = df[df['Year'] == 2]
    preprocess_train_df = df_preprocessed[df_preprocessed['Year'] == 1]
    preprocess_test_df = df_preprocessed[df_preprocessed['Year'] == 2]

    model = train_model(preprocess_train_df)
    generate_excel(train_df, preprocess_train_df, model, constants.SA_TRAIN_1)
    generate_excel(test_df, preprocess_test_df, model, constants.SA_TEST_2)

    # Generate Test 3 Predictions
    train_df = df[df['Year'] < 3]
    test_df = df[df['Year'] == 3]
    preprocess_train_df = df_preprocessed[df_preprocessed['Year'] < 3]
    preprocess_test_df = df_preprocessed[df_preprocessed['Year'] == 3]

    model = train_model(preprocess_train_df)
    generate_excel(train_df, preprocess_train_df, model, constants.SA_TRAIN_1_2)
    generate_excel(test_df, preprocess_test_df, model, constants.SA_TEST_3)
