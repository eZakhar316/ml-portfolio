# 🚗 Прогнозирование ценовой категории автомобилей<br/><br/>
## 📌 Описание проекта<br/><br/>
Анализ датасета с характеристиками автомобилей (марка, год выпуска, пробег и т.д.) и построение модели для предсказания их ценовой категории.<br/><br/>
## 📂 Данные<br/>
### Датасет содержит:<br/>
&emsp;•	Целевая переменная: price_category (цена автомобиля).<br/>
&emsp;•	Признаки: odemeter, year, и др.<br/><br/>
## 🛠 Технологии<br/>
&emsp;•	Python 3<br/>
&emsp;•	Библиотеки:<br/>
&emsp;&emsp;o	Pandas (анализ данных),<br/>
&emsp;&emsp;o	Matplotlib/Missingno (визуализация),<br/>
&emsp;&emsp;o	Scikit-learn (моделирование: Logistic Regression, Random Forest, MLPClassifier).<br/>
&emsp;•	Jupyter Notebook.<br/><br/>
## 📊 Основные этапы<br/>
1.	EDA (Exploratory Data Analysis):<br/>
&emsp;&emsp;•	Анализ распределения признаков.<br/>
&emsp;&emsp;•	Поиск выбросов и аномалий.<br/>
&emsp;&emsp;•	Корреляционный анализ.<br/>
&emsp;&emsp;•	Визуализация (боксплоты, гистограммы).<br/>
2.	Предобработка данных:<br/>
&emsp;&emsp;•	Обработка пропусков.<br/>
&emsp;&emsp;•	Кодирование категориальных переменных (One-Hot Encoding).<br/>
&emsp;&emsp;•	Нормализация/стандартизация.<br/>
3.	Modeling:<br/>
&emsp;&emsp;•	Разделение на train/test.<br/>
&emsp;&emsp;•	Обучение моделей (Logistic Regression, Random Forest, MLPClassifier).<br/>
&emsp;&emsp;•	Оценка качества (accuracy score).<br/><br/>
## 📌 Результаты<br/>
•	Лучшая модель: Logistic Regression с accuracy score = 0.95.<br/>
•	Ключевые признаки, влияющие на цену: year, odometer, model.<br/>
