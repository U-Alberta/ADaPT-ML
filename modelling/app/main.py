import os
import json
from typing import Set, List
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI(
    title="Model API",
    desctiption="API endpoints for the best-performing models of each classification task.",
    version="0.1"
)

with open(os.environ['EXAMPLE_MODEL_PATH'], 'rb') as infile:
    example_model = pickle.load(infile)


class DataPointItem(BaseModel):
    table: str
    id: Set[str]


class ExampleModelResponse(BaseModel):
    table: str
    id: Set[str]
    cat: List[int]
    dog: List[int]
    bird: List[int]


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict_pv", response_model=ExampleModelResponse)
def predict_example(data_point_item: DataPointItem):
    """
    Takes a list of data point ids from a table. Looks up the appropriate features,
    then returns predictions as columns 'class_1' ... 'class_n' and binary
    values indicating the presence/absence of the class in the prediction.
    """
    json_data_point_item = jsonable_encoder(data_point_item)
    id_json = json.dumps(json_data_point_item)
    id_df = pd.read_json(id_json)
    result_df = example_model.predict(id_df)
    response_dict = {
        'table': data_point_item.table,
        'id': data_point_item.id,
        'cat': result_df.cat.tolist(),
        'dog': result_df.dog.tolist(),
        'bird': result_df.bird.tolist(),
    }
    return response_dict
