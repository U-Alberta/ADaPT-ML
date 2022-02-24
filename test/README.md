# Testing #

The purpose of this document is to provide instructions for testing ADaPT-ML on your machine.
Please **make sure you have not modified the [`.env`](../.env) file**. These tests will bring all the containers up using docker compose, test core functionality using the data for the example use case included in [`example_data`](../example_data), and will **leave all the containers running** so that you can view the test experiment runs through the MLflow UIs and/or investigate any issues using the containers' respective log files.

## Unix: Run all tests ##

If you are using MacOS / Linux on your host machine, then run these commands:

```shell
cd ADaPT-ML/
bash ./test/all-unix.sh
```

## Windows: Run all tests ##

If you suspect that the `all-unix.sh` script will work on your host machine, then feel free to use it. Otherwise, the testing procedure can be broken down into these steps: 

1. Open the Command Line in the ADaPT-ML root directory, and bring all the containers up:
```shell
docker-compose --env-file .env --profile dev up -d
```
2. Make sure that none of the containers have exited or are restarting and wait until you see that the MLflow databases are "healthy", like this:
```shell
docker-compose ps
```
```
NAME                      COMMAND                  SERVICE             STATUS              PORTS
crate-db                  "/docker-entrypoint.…"   cratedb             running             0.0.0.0:4200->4200/tcp
dp-mlflow                 "/bin/bash"              dp                  running
dp-mlflow-db              "/entrypoint.sh mysq…"   dp_db               running (healthy)   33060-33061/tcp
dp-mlflow-server          "mlflow server --bac…"   dp_web              running             0.0.0.0:5000->5000/tcp
label-studio-dev          "/bin/bash"              ls                  running
label-studio-web          "./deploy/docker-ent…"   ls_web              running             0.0.0.0:8080->8080/tcp
modelling-mlflow          "/bin/bash"              m                   running
modelling-mlflow-db       "/entrypoint.sh mysq…"   m_db                running (healthy)   33060-33061/tcp
modelling-mlflow-deploy   "/start.sh"              m_deploy            running             0.0.0.0:8088->80/tcp
modelling-mlflow-server   "mlflow server --bac…"   m_web               running             0.0.0.0:5001->5000/tcp
```
3. Run the tests one-by-one:
```shell
docker exec label-studio-dev python /test/ls-test.py
```
```shell
docker exec dp-mlflow sh -c ". ~/.bashrc && python /test/dp-test.py"
```
```shell
docker exec modelling-mlflow sh -c ". ~/.bashrc && python /test/ml-test.py"
```
```shell
docker network create test-deploy-network --subnet 192.168.2.0/24 --gateway 192.168.2.10
docker network connect --ip 192.168.2.4 test-deploy-network modelling-mlflow-deploy
docker network connect --ip 192.168.2.8 test-deploy-network modelling-mlflow
docker exec modelling-mlflow sh -c ". ~/.bashrc && python /test/deploy-test.py"
docker network disconnect test-deploy-network modelling-mlflow-deploy
docker network disconnect test-deploy-network modelling-mlflow
docker network rm test-deploy-network
```

## Optional Clean-up ##

If you wish to bring the containers down after testing, then:

```shell
docker-compose down
```
