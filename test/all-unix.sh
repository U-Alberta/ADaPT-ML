echo "STARTING ADaPT-ML AND RUNNING ALL TESTS"

docker-compose --profile dev up -d
while docker-compose ps | grep -i "starting"
  do
    echo "Waiting for MLflow databases (5 sec)..."
    sleep 5
  done

echo "Checking for exited containers..."
! docker-compose ps | grep -i "Exit"
echo "☑ Startup complete."

. ./test/ls.sh
. ./test/dp.sh
. ./test/ml.sh
. ./test/deploy.sh

echo "ALL TESTS COMPLETED SUCCESSFULLY."