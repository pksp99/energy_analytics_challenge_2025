import warnings

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

import constants

class XGB_RF_Model:
    def __init__(self, xgb_model, rf_model):
        self.xgb_model = xgb_model
        self.rf_model = rf_model
    def predict(self, df:pd.DataFrame):
        predictions = self.xgb_model.predict(df) * 0.7 + self.rf_model.predict(df) * 0.3
        return predictions

def load_df():
    train_df = pd.read_excel(constants.ESD_TRAINING)
    test_df = pd.read_excel(constants.ESD_TESTING)
    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    df = df.fillna(0)
    return df


def preprocess_df(df_i: pd.DataFrame):
    df = df_i.copy()
    return df


def train_model(df: pd.DataFrame):
    X_train = df.drop(columns=['Load', 'Year'])
    y_train = df['Load']

    xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
    xgb_model.fit(X_train, y_train)

    rf_params = {'max_depth': 20, 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 200, 'random_state':42}
    rf_model = RandomForestRegressor(**rf_params)
    rf_model.fit(X_train, y_train)

    xgb_rf_model = XGB_RF_Model(xgb_model, rf_model)
    return xgb_rf_model


def generate_excel(df: pd.DataFrame, preprocess_df: pd.DataFrame, model, file_name: str):
    X = preprocess_df.drop(columns=['Load', 'Year'])
    y = model.predict(X)
    X[constants.BY_COL] = y
    df = df.join(X[constants.BY_COL], how='left')
    df_no_na = df.dropna()
    r2 = r2_score(df_no_na['Load'], df_no_na[constants.BY_COL])
    rmse = root_mean_squared_error(df_no_na['Load'], df_no_na[constants.BY_COL])
    mae = mean_absolute_error(df_no_na['Load'], df_no_na[constants.BY_COL])
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

    print_partition("BY_Method")

    # Generate Test 1 Predictions
    train_df = df[df['Year'] == 2]
    test_df = df[df['Year'] == 1]
    preprocess_train_df = df_preprocessed[df_preprocessed['Year'] == 2]
    preprocess_test_df = df_preprocessed[df_preprocessed['Year'] == 1]

    model = train_model(preprocess_train_df)
    generate_excel(train_df, preprocess_train_df, model, constants.BY_TRAIN_2)
    generate_excel(test_df, preprocess_test_df, model, constants.BY_TEST_1)

    # Generate Test 2 Predictions
    train_df = df[df['Year'] == 1]
    test_df = df[df['Year'] == 2]
    preprocess_train_df = df_preprocessed[df_preprocessed['Year'] == 1]
    preprocess_test_df = df_preprocessed[df_preprocessed['Year'] == 2]

    model = train_model(preprocess_train_df)
    generate_excel(train_df, preprocess_train_df, model, constants.BY_TRAIN_1)
    generate_excel(test_df, preprocess_test_df, model, constants.BY_TEST_2)

    # Generate Test 3 Predictions
    train_df = df[df['Year'] < 3]
    test_df = df[df['Year'] == 3]
    preprocess_train_df = df_preprocessed[df_preprocessed['Year'] < 3]
    preprocess_test_df = df_preprocessed[df_preprocessed['Year'] == 3]

    model = train_model(preprocess_train_df)
    generate_excel(train_df, preprocess_train_df, model, constants.BY_TRAIN_1_2)
    generate_excel(test_df, preprocess_test_df, model, constants.BY_TEST_3)
