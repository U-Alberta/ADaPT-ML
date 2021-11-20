#!/bin/bash -e
echo "=== STARTING ADaPT-ML AND RUNNING ALL TESTS ==="

docker-compose --env-file .env --profile dev up -d
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

docker exec label-studio-dev python /test/ls-test.py
docker exec dp-mlflow sh -c ". ~/.bashrc && python /test/dp-test.py"
docker exec modelling-mlflow sh -c ". ~/.bashrc && python /test/ml-test.py"
docker network connect --ip 172.19.0.8 adapt-ml_m_network modelling-mlflow-deploy
docker exec modelling-mlflow sh -c ". ~/.bashrc && python /test/deploy-test.py"
docker network disconnect adapt-ml_m_network modelling-mlflow-deploy
echo "=== TESTING COMPLETE ==="
