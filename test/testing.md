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

Please be aware that ADaPT-ML has not been tested on Windows. If you suspect that the `all-unix.sh` script will work on your host machine, then feel free to use it. Otherwise, the testing procedure can be broken down into these steps: 

1. Open a shell terminal in the ADaPT-ML root directory, and bring all the containers up:
```shell
docker-compose --env-file .env --profile dev up -d
```
2. Make sure that none of the containers have exited and wait until you see that the MLflow databases are "healthy", like this:
```shell
docker-compose ps
```
```
         Name                        Command                  State                                  Ports                            
--------------------------------------------------------------------------------------------------------------------------------------
crate-db                  /docker-entrypoint.sh crat ...   Up             0.0.0.0:4200->4200/tcp,:::4200->4200/tcp, 4300/tcp, 5432/tcp
dp-mlflow                 /bin/bash                        Up                                                                         
dp-mlflow-db              /entrypoint.sh mysqld            Up (healthy)   3306/tcp, 33060/tcp, 33061/tcp                              
dp-mlflow-server          mlflow server --backend-st ...   Up             0.0.0.0:5000->5000/tcp,:::5000->5000/tcp                    
label-studio-dev          /bin/bash                        Up                                                                         
label-studio-web          ./deploy/docker-entrypoint ...   Up             0.0.0.0:8080->8080/tcp,:::8080->8080/tcp                    
modelling-mlflow          /bin/bash                        Up                                                                         
modelling-mlflow-db       /entrypoint.sh mysqld            Up (healthy)   3306/tcp, 33060/tcp, 33061/tcp                              
modelling-mlflow-deploy   /start.sh                        Up             0.0.0.0:80->80/tcp,:::80->80/tcp                            
modelling-mlflow-server   mlflow server --backend-st ...   Up             0.0.0.0:5001->5000/tcp,:::5001->5000/tcp 
```
3. Run the tests one-by-one:
```shell
docker exec label-studio-dev python /test/ls-test.py
docker exec dp-mlflow sh -c ". ~/.bashrc && python /test/dp-test.py"
docker exec modelling-mlflow sh -c ". ~/.bashrc && python /test/ml-test.py"
docker network connect --ip 172.19.0.8 adapt-ml_m_network modelling-mlflow-deploy
docker exec modelling-mlflow sh -c ". ~/.bashrc && python /test/deploy-test.py"
docker network disconnect adapt-ml_m_network modelling-mlflow-deploy
```

## Optional Clean-up ##

If you wish to bring the containers down after testing, then:

```shell
docker-compose down
```
