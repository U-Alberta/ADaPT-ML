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

docker network connect --ip 172.19.0.8 adapt-ml_m_network modelling-mlflow-deploy
docker exec modelling-mlflow sh -c ". ~/.bashrc && python /test/deploy-test.py"
docker network disconnect adapt-ml_m_network modelling-mlflow-deploy