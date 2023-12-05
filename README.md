# Домашнее задание 1
обучение модели регрессии для предсказания стоимости автомобилей, а также реализация сервиса FastAPI для применения построенной модели на новых данных.


Никита Родионов

## Стрктура репозитория:
1. f_api_main.py (основной файл FastAPI)
2. f_api.ipynb (ноутбук для генерации .json и запуска POST метода API - развертывание локально)
3. classes.py (библиотека FastAPI)
4. f_api_JS.json (.json для отработки POST метода)
5. HW1_Regression_with_inference.ipynb (pure DS)
6. .joblib & .pickle (сохраненные модели)

## Скриншот работы сервиса
![API work demonstration](/API_demo.png)

## Что сделано
1. Предобработка датафреймов
2. Визуализация метрик
3. Обучение линейных моделей
4. Фича-инжиниринг, нормализация, регуляризация данных
5. Обучение бизнес модели и веб-сервис FastAPI

## Что не сделано
В сервисе FastAPI не реализован метод @app.post("/predict_items") для работы с .csv файлами: по какой-то причине ломался датафрейм
