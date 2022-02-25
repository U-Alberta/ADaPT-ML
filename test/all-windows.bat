ECHO OFF
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
PAUSE