import os
import json
import pprint
from typing import Set, List
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI(
    title="Model API",
    desctiption="API endpoints for the best-performing models of each classification task.",
    version="0.1"
)


class ModelException(Exception):
    def __init__(self, name: str, code: str, error=Exception("No additional details.")):
        self.name = name
        self.code = code
        self.error = error
        self.msg_dict = {
            'load': """{name} did not load successfully. 
            Please check the model path environment variable and restart the container. 
            Details: {error}""",

            'predict': """{name} did not predict successfully. Please check that the requested data points have the 
            features corresponding to the features the model was trained on, and that the database is available.
            Details: {error}"""
        }

    def get_error_msg(self):
        return self.msg_dict[self.code].format(name=self.name, error=self.error)


class DataPointItem(BaseModel):
    table: List[str]
    id: Set[str]


class ExampleModelResponse(BaseModel):
    table: List[str]
    id: Set[str]
    cat: List[int]
    dog: List[int]
    bird: List[int]
    horse: List[int]
    snake: List[int]
    prob_cat: List[float]
    prob_dog: List[float]
    prob_bird: List[float]
    prob_horse: List[float]
    prob_snake: List[float]

# Try to load each model from the location given in the .env file. Don't interfere with the startup if one can't load,
# just wait and see if the client tries to get a prediction or do something with the model that failed to load.
try:
    with open(os.environ['MULTICLASS_EXAMPLE_MODEL_PATH'], 'rb') as infile:
        multiclass_example_model = pickle.load(infile)
except FileNotFoundError as err:
    print("Could not load MulticlassExampleModel: {}".format(err))
    multiclass_example_model = None
try:
    with open(os.environ['MULTILABEL_EXAMPLE_MODEL_PATH'], 'rb') as infile:
        multilabel_example_model = pickle.load(infile)
except FileNotFoundError as err:
    print("Could not load MultilabelExampleModel: {}".format(err))
    multilabel_example_model = None

loaded_models_dict = {
    'Multiclass Example Model': multiclass_example_model is not None,
    'Multilabel Example Model': multilabel_example_model is not None
}


@app.exception_handler(ModelException)
async def model_exception_handler(request: Request, exc: ModelException):
    return JSONResponse(
        status_code=418,
        content={"message": f"Oops! {exc.get_error_msg()}"},
    )


@app.get("/")
def read_root():
    return {"Hello! This is the Model API. Here is a list of all available models:": pprint.pformat(loaded_models_dict)}


@app.post("/predict_multiclass_example", response_model=ExampleModelResponse)
def predict_multiclass_example(data_point_item: DataPointItem):
    """
    Takes a list of data point ids from a table. Looks up the appropriate features,
    then returns predictions as columns 'class_1' ... 'class_n' and binary
    values indicating the presence/absence of the class in the prediction.
    """
    try:
        assert loaded_models_dict['Multiclass Example Model']
    except Exception as error:
        raise ModelException(name="ExampleModel", code="load")
    json_data_point_item = jsonable_encoder(data_point_item)
    id_json = json.dumps(json_data_point_item)
    id_df = pd.read_json(id_json, dtype={'id': str, 'table': str})

    try:
        result_df = multiclass_example_model.predict(id_df)
    except Exception as error:
        raise ModelException(name="ExampleModel", code='predict', error=error)
    response_dict = {
        'table': result_df.table.tolist(),
        'id': result_df.id.tolist(),
        'cat': result_df.cat.tolist(),
        'dog': result_df.dog.tolist(),
        'bird': result_df.bird.tolist(),
        'horse': result_df.horse.tolist(),
        'snake': result_df.snake.tolist(),
        'prob_cat': result_df.prob_cat.tolist(),
        'prob_dog': result_df.prob_dog.tolist(),
        'prob_bird': result_df.prob_bird.tolist(),
        'prob_horse': result_df.prob_horse.tolist(),
        'prob_snake': result_df.prob_snake.tolist(),
    }

    return response_dict


@app.post("/predict_multilabel_example", response_model=ExampleModelResponse)
def predict_multilabel_example(data_point_item: DataPointItem):
    """
    Takes a list of data point ids from a table. Looks up the appropriate features,
    then returns predictions as columns 'class_1' ... 'class_n' and binary
    values indicating the presence/absence of the class in the prediction.
    """
    try:
        assert loaded_models_dict['Multilabel Example Model']
    except Exception as error:
        raise ModelException(name="ExampleModel", code="load")
    json_data_point_item = jsonable_encoder(data_point_item)
    id_json = json.dumps(json_data_point_item)
    id_df = pd.read_json(id_json, dtype={'id': str, 'table': str})
    try:
        result_df = multilabel_example_model.predict(id_df)
    except Exception as error:
        raise ModelException(name="ExampleModel", code='predict', error=error)
    response_dict = {
        'table': result_df.table.tolist(),
        'id': result_df.id.tolist(),
        'cat': result_df.cat.tolist(),
        'dog': result_df.dog.tolist(),
        'bird': result_df.bird.tolist(),
        'horse': result_df.horse.tolist(),
        'snake': result_df.snake.tolist(),
        'prob_cat': result_df.prob_cat.tolist(),
        'prob_dog': result_df.prob_dog.tolist(),
        'prob_bird': result_df.prob_bird.tolist(),
        'prob_horse': result_df.prob_horse.tolist(),
        'prob_snake': result_df.prob_snake.tolist(),
    }

    return response_dict
