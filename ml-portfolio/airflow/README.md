##📌 Описание проекта<br/>
Мониторинг ML-модели в Airflow и записи результатов работы.<br/><br/>

##🔍 Этапы работы<br/>
&emsp;1. pipeline/ содержит:<br/>
&emsp;&emsp;o	Автоматизированную обработку данных<br/>
&emsp;&emsp;o	Обучение модели<br/>
&emsp;&emsp;o	Экспорт модели в .dill формат<br/>
&emsp;2. predict/ содеожит:<br/>
&emsp;&emsp;o Загрузку лучшей модели<br/>
&emsp;&emsp;o Сохранение резудьтатов работы в .json-формате<br/>
&emsp;3. hw_dag/ содержит:<br/>
&emsp;&emsp;o Таймер повторного запуска<br/>
&emsp;&emsp;o План выполнения задач<br/><br/>

##🛠 Технологии<br/>
&emsp;•	Анализ: Pandas<br/>
&emsp;•	Моделирование: Scikit-learn, Random Forest, Logistic Regression, SVC<br/>
&emsp;•	Инфраструктура: Docker, Git<br/><br/>

##📊 Результаты<br/>
![Результат работы Airflow](https://github.com/user-attachments/assets/61e9f266-0071-4bf4-a035-1565e893496f)

