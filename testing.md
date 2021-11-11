# Testing #

The purpose of this document is to provide instructions for testing ADaPT-ML on your machine.

## Step 1: Spin up the containers using docker-compose ##

```shell
cd ADaPT-ML/
docker-compose --profile dev up -d
docker-compose ps
```
Once you see Docker Compose report this:
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
Then the containers are ready.

## Step 2: Run the test scripts ##

