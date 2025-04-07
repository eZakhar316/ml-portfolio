## 🚗 Прогнозирование целевого действия на сайте аренды автомобилей


-----------------------------------------------------------------------------

## 📌 Описание проекта

Полный цикл Data Science проекта: Pandas (EDA), Scikit-learn (Modeling), FastAPI (деплой) для предсказания конверсии в аренде авто.

--------------------------------------------------------------------------------

## 🔍 Этапы работы

### 1. EDA и Modeling в Jupyter Notebook

•	notebooks/EDA_and_Modeling.ipynb:

&emsp;o	Анализ распределения признаков

&emsp;o	Обработка пропусков и выбросов 
  
&emsp;o	Визуализация ключевых метрик
  
&emsp;o	Тестирование моделей (Random Forest, Logistic Regression, MLP Classifier, XGBoost, LightGBM, CatBoost Classifier)
  
&emsp;o	Подбор гиперпараметров
  
&emsp;o	Оценка метрик (Accuracy, ROC AUC)

### 2. Пайплайны в PyCharm

•	pipeline/ содержит:

&emsp;o	Автоматизированную обработку данных 

&emsp;o	Обучение модели 

&emsp;o	Экспорт модели в .pkl формат

### 3. FastAPI сервис

•	api/main.py:

&ensp;o	REST API с эндпоинтами:

&emsp;	/predict - получение предсказания

&emsp;	/status - статус сервиса

&emsp;	/version - версия сервиса

-------------------------------------------------------------------------------------------
## 🛠 Технологии

•	Анализ: Pandas, NumPy, Matplotlib, Seaborn, Missingno

•	Моделирование: Scikit-learn, XGBoost, LightGBM, CatBoost Classifier

•	API: FastAPI, Uvicorn

-------------------------------------------------------------------------------------------

## 📊 Результаты

|   Модель           |	Accuracy	| ROC AUC     |
|--------------------|------------|------------|
|Random Forest     	|   0.97    |  0.86       |
|Logistic Regression	|   0.97    |	 0.70      |
|MLP Classifier      |   0.97    |  0.81       |
|XGBoost             |   0.98    |  0.74       |
|LightGBM            |   0.98    |  0.75       |
|CatBoost Classifier |   0.98    |  0.68       |

