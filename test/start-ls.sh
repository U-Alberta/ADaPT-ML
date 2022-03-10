#!/bin/bash -e

if [ "$RUNNER_OS" = "Windows" ]; then
  docker-compose --file ./test/docker-compose-windows.yml --env-file .env --profile label up -d
else
  docker-compose --env-file .env --profile label up -d
fi

echo "Letting CrateDB start up..."
sleep 5
echo "Checking for exited containers..."
exited=$(docker-compose ps | grep "Exit")
if [ "$exited" = 0 ]
then
  echo "Check failed: some containers exited"
  exit 1
fi
echo "Startup complete."
docker-compose ps
