#!/bin/bash -e
echo "=== STARTING ADaPT-ML AND RUNNING ALL TESTS ==="

docker-compose --env-file .env --profile dev up -d
while docker-compose ps | grep -i "starting"
  do
    echo "Waiting for MLflow databases (5 sec)..."
    sleep 5
  done

echo "Checking for exited or restarting containers..."
exited=$(docker-compose ps | grep "Exit")
restarting=$(docker-compose ps | grep "Restarting")
if [ "$exited" = 0 ]
then
  echo "Check failed: some containers exited."
  exit 1
fi
if [ "$restarting" = 0 ]
then
  echo "Check failed: some containers restarting. Did the CrateDB bootstrap checks fail?"
  exit 1
fi
echo "Startup complete."

docker exec label-studio-dev python /test/ls-test.py
docker exec dp-mlflow sh -c ". ~/.bashrc && python /test/dp-test.py"
docker exec modelling-mlflow sh -c ". ~/.bashrc && python /test/ml-test.py"

docker network create test-deploy-network --subnet 192.168.2.0/24 --gateway 192.168.2.10
docker network connect --ip 192.168.2.4 test-deploy-network modelling-mlflow-deploy
docker network connect --ip 192.168.2.8 test-deploy-network modelling-mlflow

docker exec modelling-mlflow sh -c ". ~/.bashrc && python /test/deploy-test.py"

docker network disconnect test-deploy-network modelling-mlflow-deploy
docker network disconnect test-deploy-network modelling-mlflow
docker network rm test-deploy-network
echo "=== TESTING COMPLETE ==="
