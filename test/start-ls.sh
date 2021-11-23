#!/bin/bash -e

docker-compose --env-file .env --profile label up -d

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

docker exec label-studio-dev python /test/ls-test.py
