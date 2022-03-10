#!/bin/bash -e

if [ "$RUNNER_OS" = "Windows" ]; then
  docker-compose --file ./test/docker-compose-windows.yml --env-file .env --profile deploy up -d
else
  docker-compose --env-file .env --profile deploy up -d
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

#if [ "$RUNNER_OS" == "Linux" ]; then
#              apt install important_linux_software
#         elif [ "$RUNNER_OS" == "Windows" ]; then
#              choco install important_windows_software
#         else
#              echo "$RUNNER_OS not supported"
#              exit 1
#         fi
# set up a temporary network to use to test the model deployment
docker network create test-deploy-network --subnet 192.168.2.0/24 --gateway 192.168.2.10
docker network connect --ip 192.168.2.4 test-deploy-network modelling-mlflow-deploy
docker network connect --ip 192.168.2.8 test-deploy-network modelling-mlflow

docker exec modelling-mlflow sh -c ". ~/.bashrc && python /test/deploy-test.py"

docker network disconnect test-deploy-network modelling-mlflow-deploy
docker network disconnect test-deploy-network modelling-mlflow
docker network rm test-deploy-network
