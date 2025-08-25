#!/bin/bash

echo "Stopping services..."
docker compose down

echo "Starting Redis (needed for app initialization)..."
docker compose up -d redis

echo "Running database migration..."
docker compose run --rm web python /app/migrate_db.py

echo "Starting all services..."
docker compose up -d

echo "Database fix completed!"
echo "You can now try deleting slideshows again."