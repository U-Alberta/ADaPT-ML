#!/bin/bash -e

docker-compose --env-file .env --profile deploy up -d
while docker-compose ps | grep -i "starting"
  do
    echo "Waiting for MLflow databases (5 sec)..."
    sleep 5
  done

echo "Checking for exited containers..."
exited=$(docker-compose ps | grep "Exit")
if [ "$exited" = 0 ]
then
  echo "Check failed: some containers exited"
  exit 1
fi
echo "Startup complete."

# set up a temporary network to use to test the model deployment
docker network create test-deploy-network
docker network connect --ip 172.19.0.8 test-deploy-network modelling-mlflow-deploy
docker network connect --ip 172.19.0.4 test-deploy-network modelling-mlflow

docker exec modelling-mlflow sh -c ". ~/.bashrc && python /test/deploy-test.py"

docker network disconnect test-deploy-network modelling-mlflow-deploy
docker network disconnect test-deploy-network modelling-mlflow
docker network rm test-deploy-network
