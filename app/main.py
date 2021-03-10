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

with open(os.environ['PV_MODEL_PATH'], 'rb') as infile:
    pv_model = pickle.load(infile)


class TweetIDItem(BaseModel):
    table: str
    ids: Set[str]


class PVModelResponse(BaseModel):
    table: str
    ids: Set[str]
    security: List[int]
    conformity: List[int]
    tradition: List[int]
    benevolence: List[int]
    universalism: List[int]
    self_direction: List[int]
    stimulation: List[int]
    hedonism: List[int]
    achievement: List[int]
    power: List[int]


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict_pv", response_model=PVModelResponse)
def predict_pv(tweet_id_item: TweetIDItem):
    """
    Takes a list of data point ids from a table. Looks up the appropriate features,
    then returns predictions as columns 'class_1' ... 'class_n' and binary
    values indicating the presence/absence of the class in the prediction.
    """
    json_tweet_id_item = jsonable_encoder(tweet_id_item)
    id_json = json.dumps(json_tweet_id_item)
    id_df = pd.read_json(id_json)
    result_df = pv_model.predict(id_df)
    response_dict = {
        'table': tweet_id_item.table,
        'ids': tweet_id_item.ids,
        'security': result_df.security.tolist(),
        'conformity': result_df.conformity.tolist(),
        'tradition': result_df.tradition.tolist(),
        'benevolence': result_df.benevolence.tolist(),
        'universalism': result_df.universalism.tolist(),
        'self_direction': result_df.self_direction.tolist(),
        'stimulation': result_df.stimulation.tolist(),
        'hedonism': result_df.hedonism.tolist(),
        'achievement': result_df.achievement.tolist(),
        'power': result_df.power.tolist(),
    }
    return response_dict
