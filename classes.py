from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from category_encoders import TargetEncoder


class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


class MyModel:

    def __init__(self):
        self.scaler = joblib.load("scaler_model.joblib")
        self.ohe = joblib.load("ohe_model.joblib")
        self.encoder = joblib.load("encoder_model.joblib")
        self.model = joblib.load("model.joblib")

    def preprocess_data(self, data):

        data['mileage'] = pd.to_numeric(data['mileage'].str.replace(' kmpl', '').str.replace(' km/kg', ''),
                                        errors='coerce')
        data['engine'] = pd.to_numeric(data['engine'].str.replace(' CC', ''), errors='coerce')
        data['max_power'] = pd.to_numeric(data['max_power'].str.replace(' bhp', ''), errors='coerce')
        data['power_per_seat'] = data['max_power'] / (data['seats'] + 1e-6)
        data['mileage_to_power_ratio'] = data['mileage'] / (data['max_power'] + 1e-6)
        data['year_to_km_driven_ratio'] = data['year'] / (data['km_driven'] + 1e-6)
        data['max_power_deviarion'] = data['max_power'] - data['max_power'].mean()
        def rank_code(row):
            if row == 'Test Drive Car':
                return 0
            elif row == 'First Owner':
                return 1
            elif row == 'Second Owner':
                return 2
            elif row == 'Third Owner':
                return 3
            else:
                return 4

        data['owner'] = data['owner'].map(rank_code)
        numeric = data[
            ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'power_per_seat', 'mileage_to_power_ratio',
             'year_to_km_driven_ratio', 'max_power_deviarion', 'name']].columns
        categorical = data[['fuel', 'transmission', 'seller_type']].columns
        data['name'] = self.encoder.transform(data['name'])
        data[numeric] = self.scaler.transform(data[numeric])
        squared_features = data[numeric].apply(lambda x: x ** 2)
        data = pd.concat([data, squared_features.add_suffix('_squared')], axis=1)
        ohe_data = self.ohe.transform(data[categorical]).toarray()
        ohe_data = pd.DataFrame(ohe_data, columns=self.ohe.get_feature_names_out(categorical))
        result_data = data = pd.concat([data, ohe_data], axis=1).drop(categorical, axis=1)
        return result_data

    def predict(self, data):
        preprocessed_data = self.preprocess_data(data)
        return self.model.predict(preprocessed_data)





