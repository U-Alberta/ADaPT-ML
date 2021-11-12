docker-compose --profile deploy up -d

echo "Checking for exited containers..."
! docker-compose ps | grep -i "Exit"
echo "☑ Startup complete."