import os
import json
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
        return self.msg_dict[self.code].format(name=self.name, err=self.error)


class DataPointItem(BaseModel):
    table: str
    id: Set[str]


class ExampleModelResponse(BaseModel):
    table: str
    id: Set[str]
    cat: List[int]
    dog: List[int]
    bird: List[int]


# Try to load each model from the location given in the .env file. Don't interfere with the startup if one can't load,
# just wait and see if the client tries to get a prediction or do something with the model that failed to load.
try:
    with open(os.environ['EXAMPLE_MODEL_PATH'], 'rb') as infile:
        example_model = pickle.load(infile)
except FileNotFoundError as err:
    print("Could not load ExampleModel: {}".format(err))
    example_model = None


@app.exception_handler(ModelException)
async def model_exception_handler(request: Request, exc: ModelException):
    return JSONResponse(
        status_code=418,
        content={"message": f"Oops! {exc.get_error_msg()}"},
    )


@app.get("/")
def read_root():
    return {"Hello!": "This is the Model API."}


@app.post("/predict_example", response_model=ExampleModelResponse)
def predict_example(data_point_item: DataPointItem):
    """
    Takes a list of data point ids from a table. Looks up the appropriate features,
    then returns predictions as columns 'class_1' ... 'class_n' and binary
    values indicating the presence/absence of the class in the prediction.
    """
    if example_model is None:
        raise ModelException(name="ExampleModel", code="load")
    json_data_point_item = jsonable_encoder(data_point_item)
    id_json = json.dumps(json_data_point_item)
    id_df = pd.read_json(id_json)
    try:
        result_df = example_model.predict(id_df)
    except Exception as error:
        raise ModelException(name="ExampleModel", code='predict', error=error)
    response_dict = {
        'table': data_point_item.table,
        'id': data_point_item.id,
        'cat': result_df.cat.tolist(),
        'dog': result_df.dog.tolist(),
        'bird': result_df.bird.tolist(),
    }

    return response_dict
