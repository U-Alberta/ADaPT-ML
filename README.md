# CECN Data Programming

Repo for creating training data through heuristics and external knowledge, as well as manually annotating small samples of data. Right now, we are using the Personal Values Dictionary developed by Ponizovskiy et al. (2020) to automatically create training data for
modelling the personal values people are expressing in text.

![Data programming and classification flow.](/graphics/dp_class_flow.png "dp and class flow")

https://towardsdatascience.com/targeted-sentiment-analysis-vs-traditional-sentiment-analysis-4d9f2c12a476
https://github.com/songyouwei/ABSA-PyTorch
https://hub.docker.com/_/mysql?tab=description

## Setup

```shell script
cd $DP_REPO_PATH
docker-compose up -d --build
```
*Note:* make sure that you export values for the environment variables in docker-compose.yml

Will create and start the MySQL backend store, the MLFlow server (start it manually for now though, see below), and the
interface for running MLFlow experiments. Will also start the Label Studio server for annotating development / test data.

```shell script
docker attach dp-mlflow-server
wait-for-it dp-mlflow-db:3306 -s -- mlflow server --backend-store-uri $MLFLOW_TRACKING_URI --default-artifact-root ./mlruns --host 0.0.0.0
```
Will start the MLFlow server manually

## Main Usage

```shell script
docker attach dp-mlflow
wait-for-it dp-mlflow-db:3306 -s -- mlflow run --no-conda -e <pv> --experiment-name <energyeast> -P data=/unlabeled_data</path/to/data.pkl> .
```
or
```shell script
docker exec dp-mlflow 'wait-for-it dp-mlflow-db:3306 -s -- mlflow run --no-conda -e <entrypoint> --experiment-name <name> -P data=/unlabeled_data</path/to/data.pkl> .'
```
*-e pv*: the entrypoint is for assessing personal values

*-P pv_data*: path to the unlabeled training data

Go to http://129.128.215.241:5000 to view the experiments

## Labeling data with Label Studio

after running the [Setup](#Setup), do
```shell script
docker attach label-studio
label-studio init --input-path=./tasks --input-format=json-dir
```
or
```shell script
docker exec label-studio 'label-studio init --input-path=./tasks --input-format=json-dir'
```

This will import the JSON-formatted list of data points in each file in the input path. The files should look like this:
```json
[
  {
    "tweet_text": "Opossum is great",
    "ref_id": "<tweet_id>",
    "meta_info": {
      "timestamp": "2020-03-09 18:15:28.212882",
      "location": "North Pole"
    }
  }
]
```
