import warnings

import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

import constants
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
def generate_results_latex(df: pd.DataFrame):
    for row in df.values:
        print(' & '.join(row))


def generate_excel(df: pd.DataFrame, preprocess_df: pd.DataFrame, model, file_name: str, col_name: str):
    X = preprocess_df.drop(columns=['Load', 'Year'])
    y = model.predict(X)
    X[col_name] = y
    df = df.join(X[col_name], how='left')
    df_no_na = df.dropna()
    r2 = r2_score(df_no_na['Load'], df_no_na[col_name])
    rmse = root_mean_squared_error(df_no_na['Load'], df_no_na[col_name])
    mae = mean_absolute_error(df_no_na['Load'], df_no_na[col_name])
    mape = mean_absolute_percentage_error(df_no_na['Load'], df_no_na[col_name])
    logger.info(f"{file_name:<30} {str(df_no_na.shape):<10} -> R^2: {r2:.2f} \t RMSE: {rmse:.2f} \t MAE: {mae:.2f} \t MAPE: {mape:.2%}")
    df.to_excel(file_name, index=False)


def load_df():
    train_df = pd.read_excel(constants.ESD_TRAINING)
    test_df = pd.read_excel(constants.ESD_TESTING)
    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    df = df.fillna(0)
    return df

def print_partition(title=""):
    print("\n" + "=" * 50)
    if title:
        print(f"{title.center(50)}")
        print("=" * 50)