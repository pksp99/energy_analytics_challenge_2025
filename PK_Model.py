import warnings

import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

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
    X_train = df.drop(columns=['Load'])
    y_train = df['Load']
    xgb_params = {'subsample': 0.2, 'n_estimators': 350, 'max_depth': 7, 'learning_rate': 0.017,
                  'colsample_bytree': 0.25}
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train)
    return xgb_model


def generate_excel(df: pd.DataFrame, preprocess_df: pd.DataFrame, model, file_name: str):
    X = preprocess_df.drop(columns=['Load'])
    y = model.predict(X)
    X[constants.PK_COL] = y
    df = df.join(X[constants.PK_COL], how='left')
    df_no_na = df.dropna()
    r2 = r2_score(df_no_na['Load'], df_no_na[constants.PK_COL])
    rmse = root_mean_squared_error(df_no_na['Load'], df_no_na[constants.PK_COL])
    mae = mean_absolute_error(df_no_na['Load'], df_no_na[constants.PK_COL])
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

    print_partition("PK_Method")

    # Generate Test 1 Predictions
    train_df = df[df['Year'] == 2]
    test_df = df[df['Year'] == 1]
    preprocess_train_df = df_preprocessed[df_preprocessed['Year'] == 2]
    preprocess_test_df = df_preprocessed[df_preprocessed['Year'] == 1]

    model = train_model(preprocess_train_df)
    generate_excel(train_df, preprocess_train_df, model, constants.PK_TRAIN_2)
    generate_excel(test_df, preprocess_test_df, model, constants.PK_TEST_1)

    # Generate Test 2 Predictions
    train_df = df[df['Year'] == 1]
    test_df = df[df['Year'] == 2]
    preprocess_train_df = df_preprocessed[df_preprocessed['Year'] == 1]
    preprocess_test_df = df_preprocessed[df_preprocessed['Year'] == 2]

    model = train_model(preprocess_train_df)
    generate_excel(train_df, preprocess_train_df, model, constants.PK_TRAIN_1)
    generate_excel(test_df, preprocess_test_df, model, constants.PK_TEST_2)

    # Generate Test 3 Predictions
    train_df = df[df['Year'] < 3]
    test_df = df[df['Year'] == 3]
    preprocess_train_df = df_preprocessed[df_preprocessed['Year'] < 3]
    preprocess_test_df = df_preprocessed[df_preprocessed['Year'] == 3]

    model = train_model(preprocess_train_df)
    generate_excel(train_df, preprocess_train_df, model, constants.PK_TRAIN_1_2)
    generate_excel(test_df, preprocess_test_df, model, constants.PK_TEST_3)
