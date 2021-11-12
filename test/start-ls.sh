docker-compose --profile label up -d

echo "Checking for exited containers..."
! docker-compose ps | grep -i "Exit"
echo "☑ Startup complete."
