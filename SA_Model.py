import warnings

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

import methods

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

import constants
from collections import defaultdict


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





import multiprocessing

def generate_predictions(test_case, df, df_preprocessed):

    if test_case == 1:
        # Generate Test 1 Predictions
        train_df = df[df['Year'] == 2]
        test_df = df[df['Year'] == 1]
        preprocess_train_df = df_preprocessed[df_preprocessed['Year'] == 2]
        preprocess_test_df = df_preprocessed[df_preprocessed['Year'] == 1]

        model = train_model(preprocess_train_df)
        methods.generate_excel(train_df, preprocess_train_df, model, constants.SA_TRAIN_2, constants.SA_COL)
        methods.generate_excel(test_df, preprocess_test_df, model, constants.SA_TEST_1, constants.SA_COL)

    elif test_case == 2:
        # Generate Test 2 Predictions
        train_df = df[df['Year'] == 1]
        test_df = df[df['Year'] == 2]
        preprocess_train_df = df_preprocessed[df_preprocessed['Year'] == 1]
        preprocess_test_df = df_preprocessed[df_preprocessed['Year'] == 2]

        model = train_model(preprocess_train_df)
        methods.generate_excel(train_df, preprocess_train_df, model, constants.SA_TRAIN_1, constants.SA_COL)
        methods.generate_excel(test_df, preprocess_test_df, model, constants.SA_TEST_2, constants.SA_COL)

    elif test_case == 3:
        # Generate Test 3 Predictions
        train_df = df[df['Year'] < 3]
        test_df = df[df['Year'] == 3]
        preprocess_train_df = df_preprocessed[df_preprocessed['Year'] < 3]
        preprocess_test_df = df_preprocessed[df_preprocessed['Year'] == 3]

        model = train_model(preprocess_train_df)
        methods.generate_excel(train_df, preprocess_train_df, model, constants.SA_TRAIN_1_2, constants.SA_COL)
        methods.generate_excel(test_df, preprocess_test_df, model, constants.SA_TEST_3, constants.SA_COL)

def main():
    df = methods.load_df()
    df_preprocessed = preprocess_df(df)

    methods.print_partition("SA_Method")

    # Create a list of test cases to process in parallel
    test_cases = [1, 2, 3]

    # Using multiprocessing to run predictions in parallel
    with multiprocessing.Pool(processes=len(test_cases)) as pool:
        pool.starmap(generate_predictions, [(test_case, df, df_preprocessed) for test_case in test_cases])

if __name__ == "__main__":
    main()
