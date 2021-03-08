from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()
model_path = '/mlruns/1/df8e5c1c720e4376b1d30faf7b060e84/artifacts/mlp/python_model.pkl'

with open(model_path, 'rb') as infile:
    model = pickle.load(infile)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def predict(id_df: pd.DataFrame):
    result_df = model.predict(id_df)
    return result_df.to_json()
