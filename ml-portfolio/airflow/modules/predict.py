import os
import dill
import pandas as pd
import json
from datetime import datetime

# Укажем путь к файлам проекта:
# -> $PROJECT_PATH при запуске в Airflow
# -> иначе - текущая директория при локальном запуске
path = os.environ.get('PROJECT_PATH', '/opt/airflow/plugins')


def load_model():
    models_dir = f'{path}/data/models'
    model_files = os.listdir(models_dir)
    model_files.sort()
    latest_model = model_files[-1]
    with open(os.path.join(models_dir, latest_model), 'rb') as file:
        model = dill.load(file)
    return model


def predict():
    model = load_model()

    test_dir = f'{path}/data/test'
    predictions = []

    for filename in os.listdir(test_dir):
        if filename.endswith('.json'):
            with open(os.path.join(test_dir, filename), 'r') as file:
                json_data = json.load(file)
                test_data = pd.DataFrame([json_data])
                if 'price_category' in test_data.columns:
                    X = test_data.drop('price_category', axis=1)
                else:
                    X = test_data
                y_pred = model.predict(X)
                predictions.append(pd.DataFrame({
                    'id': test_data['id'],
                    'price_category': y_pred
                }))


    predictions_df = pd.concat(predictions, ignore_index=True)
    predictions_df.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)


if __name__ == '__main__':
    predict()