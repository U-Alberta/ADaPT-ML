# CECN Data Programming

Using the Personal Values Dictionary developed by Ponizovskiy et al. (2020) to automatically create training data for
modelling the personal values people are expressing in text.

https://towardsdatascience.com/targeted-sentiment-analysis-vs-traditional-sentiment-analysis-4d9f2c12a476
https://github.com/songyouwei/ABSA-PyTorch
https://hub.docker.com/_/mysql?tab=description

## Setup

```shell script
cd cecn-data-programming
docker-compose up -d --build
```
Will create and start the MySQL backend store, the MLFlow server (start it manually for now though, see below), and the
interface for running MLFlow experiments.

```shell script
docker attach mlflow_server
wait-for-it mlflow_db:3306 -s -- mlflow server --backend-store-uri mysql+pymysql://mlflow:superstrongpassword@$mlflow_db:3306/mlruns.db --default-artifact-root ./mlruns --host 0.0.0.0
```
Will start the server manually

*Note:* make sure that there is a .env file in the project directory that contains values for the environment variables in docker-compose.yml

## Main Usage

```shell script
docker attach mlflow_label
wait-for-it mlflow_db:3306 -s -- mlflow run --no-conda -e pv --experiment-name energyeast -P pv_data=./label/data/energyeastframe.pkl .
```
*-e pv*: the entrypoint is for assessing personal values

*-P pv_data*: path to the unlabeled training data

Go to http://206.12.89.194:5000 to view the experiments
