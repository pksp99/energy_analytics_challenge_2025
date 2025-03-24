import warnings

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

import methods

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

import constants

class XGB_RF_Model:
    def __init__(self, xgb_model, rf_model):
        self.xgb_model = xgb_model
        self.rf_model = rf_model
    def predict(self, df:pd.DataFrame):
        predictions = self.xgb_model.predict(df) * 0.7 + self.rf_model.predict(df) * 0.3
        return predictions



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






if __name__ == "__main__":
    df = methods.load_df()
    df_preprocessed = preprocess_df(df)

    methods.print_partition("BY_Method")

    # Generate Test 1 Predictions
    train_df = df[df['Year'] == 2]
    test_df = df[df['Year'] == 1]
    preprocess_train_df = df_preprocessed[df_preprocessed['Year'] == 2]
    preprocess_test_df = df_preprocessed[df_preprocessed['Year'] == 1]

    model = train_model(preprocess_train_df)
    methods.generate_excel(train_df, preprocess_train_df, model, constants.BY_TRAIN_2, constants.BY_COL)
    methods.generate_excel(test_df, preprocess_test_df, model, constants.BY_TEST_1, constants.BY_COL)

    # Generate Test 2 Predictions
    train_df = df[df['Year'] == 1]
    test_df = df[df['Year'] == 2]
    preprocess_train_df = df_preprocessed[df_preprocessed['Year'] == 1]
    preprocess_test_df = df_preprocessed[df_preprocessed['Year'] == 2]

    model = train_model(preprocess_train_df)
    methods.generate_excel(train_df, preprocess_train_df, model, constants.BY_TRAIN_1, constants.BY_COL)
    methods.generate_excel(test_df, preprocess_test_df, model, constants.BY_TEST_2, constants.BY_COL)

    # Generate Test 3 Predictions
    train_df = df[df['Year'] < 3]
    test_df = df[df['Year'] == 3]
    preprocess_train_df = df_preprocessed[df_preprocessed['Year'] < 3]
    preprocess_test_df = df_preprocessed[df_preprocessed['Year'] == 3]

    model = train_model(preprocess_train_df)
    methods.generate_excel(train_df, preprocess_train_df, model, constants.BY_TRAIN_1_2, constants.BY_COL)
    methods.generate_excel(test_df, preprocess_test_df, model, constants.BY_TEST_3, constants.BY_COL)
