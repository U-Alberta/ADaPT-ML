#!/bin/bash -e

if [ "$RUNNER_OS" = "Windows" ]; then
  docker-compose --file ./test/docker-compose-windows.yml --env-file .env --profile modelling up -d
else
  docker-compose --env-file .env --profile modelling up -d
fi

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

docker exec modelling-mlflow sh -c ". ~/.bashrc && python /test/ml-test.py"
