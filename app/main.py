from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()
model_path = '/mlruns/1/66f85b78775344d1afb8db53bb0eab6c/artifacts/mlp/python_model.pkl'

with open(model_path, 'rb') as infile:
    model = pickle.load(infile)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def predict(id_json: str):
    """
    Takes a DataFrame converted to a json string with columns 'id' and 'table'. Looks up the appropriate features,
    then returns predictions as a DataFrame converted to a json string with columns 'class_1' ... 'class_n' and binary
    values indicating the presence/absence of the class in the prediction.
    """
    id_df = pd.read_json(id_json)
    result_df = model.predict(id_df)
    return result_df.to_json()
