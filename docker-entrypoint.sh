#!/bin/bash
set -e

echo "Photo-Framer startup script"
echo "============================"

# Run database migrations
echo "Checking for pending database migrations..."
if [ -f "migrations/add_per_link_weather.py" ]; then
    echo "Running per-link weather migration..."
    python3 migrations/add_per_link_weather.py migrate || echo "Migration already applied or failed"
fi

echo "Migrations complete!"
echo ""

# Start the application
echo "Starting Photo-Framer application..."
exec "$@"
