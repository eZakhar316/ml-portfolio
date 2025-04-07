import pandas as pd
import numpy as np
import dill
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from datetime import datetime
import logging
from functools import partial



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_target_actions(ga_sessions, ga_hits):
    target_actions = [
        'sub_car_claim_click', 'sub_car_claim_submit_click', 'sub_open_dialog_click',
        'sub_custom_question_submit_click', 'sub_call_number_click', 'sub_callback_submit_click',
        'sub_submit_success', 'sub_car_request_submit_click'
    ]
    ga_hits['target_action'] = ga_hits['event_action'].isin(target_actions).astype(int)
    target_df = ga_hits.groupby('session_id')['target_action'].sum().reset_index()
    target_df['target_action'] = (target_df['target_action'] > 0).astype(int)
    ga_sessions = ga_sessions.merge(target_df, on='session_id', how='left')
    ga_sessions['target_action'] = ga_sessions['target_action'].fillna(0).astype(int)
    return ga_sessions



def first_drop(ga_sessions):
    columns_to_drop = [
        'utm_keyword',
        'device_os',
        'device_model'
    ]
    return ga_sessions.drop(columns_to_drop, axis=1)


def second_drop(ga_sessions):
    columns_to_drop = [
        'client_id',
        'visit_date',
        'visit_time',
        'visit_number'
    ]
    return ga_sessions.drop(columns_to_drop, axis=1)


def delete_nan(ga_sessions):
    columns_to_check = ['utm_source', 'utm_campaign', 'utm_adcontent', 'device_brand']
    log_missing_values(ga_sessions, "before delete_nan")
    ga_sessions = ga_sessions.dropna(subset=columns_to_check)
    log_missing_values(ga_sessions, "after delete_nan")
    return ga_sessions.reset_index(drop=True)


def process_screen_resolutions(ga_sessions):
    def check_resolution_format(resolution):
        if pd.isna(resolution):
            return False
        parts = resolution.split('x')
        return len(parts) == 2 and all(part.isdigit() for part in parts)

    def split_resolution(resolution):
        width, height = resolution.split('x')
        return int(width), int(height)

    ga_sessions['valid_resolution'] = ga_sessions['device_screen_resolution'].apply(check_resolution_format)
    ga_sessions = ga_sessions[ga_sessions['valid_resolution']].copy().reset_index(drop=True)
    ga_sessions[['device_screen_width', 'device_screen_height']] = (
        ga_sessions['device_screen_resolution']
        .apply(lambda x: pd.Series(split_resolution(x)))
    )
    ga_sessions.drop(columns=['device_screen_resolution', 'valid_resolution'], inplace=True)
    return ga_sessions


def remove_outliers_screen(ga_sessions):
    def calculate_outliers(data):
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        iqr = q75 - q25
        boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)
        return boundaries

    columns = ['device_screen_width', 'device_screen_height']
    for column in columns:
        boundaries = calculate_outliers(ga_sessions[column])
        ga_sessions = ga_sessions[
            (ga_sessions[column] >= boundaries[0]) & (ga_sessions[column] <= boundaries[1])
        ].reset_index(drop=True)
    return ga_sessions


def log_shape(data):
    logging.info(f"Data shape after step: {data.shape}")
    return data


def log_missing_values(df, step_name):
    missing_values = df.isnull().sum()
    logging.info(f"Missing values after '{step_name}':\n{missing_values}")


numeric_features = make_column_selector(dtype_include=['int64', 'float64'])
numeric_transformer = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler()
)

categorical_features = make_column_selector(dtype_include=object)
categorical_transformer = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder()
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)


def main():
    logging.info("Loading data...")
    ga_sessions = pd.read_csv('data/ga_sessions.csv', low_memory=False)
    ga_hits = pd.read_csv('data/ga_hits.csv')
    logging.info("Data loaded.")


    logging.info("Sample indices...")
    sample_indices = np.random.choice(ga_sessions.shape[0], size=int(ga_sessions.shape[0] * 0.1), replace=False)
    ga_sessions = ga_sessions.iloc[sample_indices].reset_index(drop=True)
    logging.info("Done.")

    create_target_actions_with_hits = partial(create_target_actions, ga_hits=ga_hits)


    preprocessing_pipeline = Pipeline(steps=[
        ('first_drop', FunctionTransformer(first_drop)),
        ('log_filter_1', FunctionTransformer(log_shape)),  # Переименован
        ('remove_nans', FunctionTransformer(delete_nan)),
        ('log_remove_nans', FunctionTransformer(log_shape)),
        ('process_resolutions', FunctionTransformer(process_screen_resolutions)),
        ('log_process_resolutions', FunctionTransformer(log_shape)),
        ('adding_target', FunctionTransformer(create_target_actions_with_hits)),
        ('log_target', FunctionTransformer(log_shape)),
        ('remove_outliers', FunctionTransformer(remove_outliers_screen)),
        ('log_remove_outliers', FunctionTransformer(log_shape)),
        ('second_drop', FunctionTransformer(second_drop)),
        ('log_filter_2', FunctionTransformer(log_shape))  # Переименован
    ])

    logging.info("Processing data...")
    ga_sessions = preprocessing_pipeline.fit_transform(ga_sessions)
    logging.info("Done.")


    logging.info("Processing data split...")
    x = ga_sessions.drop(['target_action'], axis=1)
    y = ga_sessions['target_action']
    logging.info("Done.")


    logging.info("Delete 'sessions_id'...")
    x = ga_sessions.drop(['session_id'], axis=1)
    logging.info("Done.")



    x_preprocessed = preprocessor.fit_transform(x)



    models = (
        RandomForestClassifier(),
        LogisticRegression(),
        MLPClassifier(random_state=42, max_iter=500),
        xgb.XGBClassifier(eval_metric='logloss', tree_method='hist'),
        lgb.LGBMClassifier(boosting_type='gbdt', n_estimators=100),
        CatBoostClassifier(iterations=100, learning_rate=0.1)
    )

    best_model = None
    best_score = 0
    for model in models:
        logging.info(f"Training model: {type(model).__name__}")
        pipe = Pipeline(steps=[
            ('classifier', model)
        ])
        cv_score = cross_val_score(pipe, x_preprocessed, y, cv=5)
        y_pred_proba = pipe.fit(x_preprocessed, y).predict_proba(x_preprocessed)[:, 1]
        roc_auc = roc_auc_score(y, y_pred_proba)
        logging.info(f'model: {type(model).__name__}, ROC AUC: {roc_auc:.4f}, cv_score_mean: {cv_score.mean():.4f}, cv_score_std: {cv_score.std():.4f}')
        if roc_auc > best_score:
            best_score = roc_auc
            best_model = model

    logging.info(f'Best model: {type(best_model).__name__}, Best ROC AUC: {best_score:.4f}')


    with open('target_pipe.pkl', 'wb') as f:
        dill.dump({
            'model': best_model,
            'preprocessor': preprocessor,
            'metadata': {
                'name': 'Target prediction pipeline',
                'author': 'Ekaterina Zakharova',
                'version': 1,
                'data': datetime.now(),
                'type': type(best_model).__name__,
                'roc auc': best_score
            }
        }, f)
    logging.info("Model saved.")


if __name__ == '__main__':
    main()