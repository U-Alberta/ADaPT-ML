#!/bin/bash -e

docker-compose --env-file .env --profile data_programming up -d
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
docker-compose ps

docker exec dp-mlflow sh -c ". ~/.bashrc && python /test/dp-test.py"
