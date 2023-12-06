from fastapi import FastAPI
from classes import Item
from classes import Items
from classes import MyModel
from typing import List
import pandas as pd


app = FastAPI()

model_instance = MyModel()

@app.get('/')
def root():
    return 'OK'


@app.post("/predict_item")
def predict_item(item: Item) -> float:

    input_data = pd.DataFrame([item.dict()])
    prediction = model_instance.predict(input_data)

    return prediction[0]