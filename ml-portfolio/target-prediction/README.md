# 🚗 Прогнозирование целевого действия на сайте аренды автомобилей<br/><br/>
## 📌 Описание проекта<br/>
Полный цикл Data Science проекта: Pandas (EDA), Scikit-learn (Modeling), FastAPI (деплой) для предсказания целевого действия на сайте аренды авто.<br/><br/>
## 🔍 Этапы работы<br/>
### 1. EDA и Modeling в Jupyter Notebook<br/>
•	notebooks/EDA_and_Modeling.ipynb:<br/>
&emsp;o	Анализ распределения признаков<br/>
&emsp;o	Обработка пропусков и выбросов<br/>
&emsp;o	Визуализация ключевых метрик<br/>  
&emsp;o	Тестирование моделей (Random Forest, Logistic Regression, MLP Classifier, XGBoost, LightGBM, CatBoost Classifier)<br/> 
&emsp;o	Подбор гиперпараметров<br/> 
&emsp;o	Оценка метрик (Accuracy, ROC AUC)<br/>
### 2. Пайплайны в PyCharm<br/>
•	pipeline/ содержит:<br/>
&emsp;o	Автоматизированную обработку данных<br/> 
&emsp;o	Обучение модели<br/> 
&emsp;o	Экспорт модели в .pkl формат<br/>
### 3. FastAPI сервис<br/>
•	api/main.py:<br/>
&ensp;o	REST API с эндпоинтами:<br/>
&emsp;✓	/predict - получение предсказания<br/>
&emsp;✓	/status - статус сервиса<br/>
&emsp;✓	/version - версия сервиса<br/><br/>
## 🛠 Технологии<br/>
•	Анализ: Pandas, NumPy, Matplotlib, Seaborn, Missingno<br/>
•	Моделирование: Scikit-learn, XGBoost, LightGBM, CatBoost Classifier<br/>
•	API: FastAPI, Uvicorn<br/><br/>
## 📊 Результаты<br/>
|   Модель           |	Accuracy	| ROC AUC     |
|--------------------|------------|------------|
|Random Forest     	|   0.97    |  0.86       |
|Logistic Regression	|   0.97    |	 0.70      |
|MLP Classifier      |   0.97    |  0.81       |
|XGBoost             |   0.98    |  0.74       |
|LightGBM            |   0.98    |  0.75       |
|CatBoost Classifier |   0.98    |  0.68       |

