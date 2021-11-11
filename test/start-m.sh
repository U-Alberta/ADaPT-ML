docker-compose --profile modelling up -d

while docker-compose ps | grep -i "starting"
  do
    echo "Waiting for MLflow database (5 sec)..."
    sleep 5
  done

echo "Checking for exited containers..."
! docker-compose ps | grep -i "Exit"
echo "☑ Startup complete."
