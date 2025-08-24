#!/bin/bash
# Photo Frame Slideshow Server Startup Script

set -e

echo "🖼️  Photo Frame Slideshow Server"
echo "================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first:"
    echo "   https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first:"
    echo "   https://docs.docker.com/compose/install/"
    exit 1
fi

# Check if we're in the right directory
if [[ ! -f "docker-compose.yml" ]]; then
    echo "❌ docker-compose.yml not found. Please run this script from the project directory."
    exit 1
fi

# Create .env file if it doesn't exist
if [[ ! -f ".env" ]]; then
    echo "📝 Creating .env configuration file..."
    cp .env.example .env
    
    # Generate a random secret key
    SECRET_KEY=$(openssl rand -hex 32 2>/dev/null || python3 -c "import secrets; print(secrets.token_hex(32))" 2>/dev/null || echo "change-this-secret-key-$(date +%s)")
    sed -i.bak "s/your-secret-key-change-this-in-production/$SECRET_KEY/" .env
    rm -f .env.bak
    
    echo "✅ .env file created with random secret key"
    echo "⚠️  IMPORTANT: Change ADMIN_PASSWORD in .env file before production use!"
fi

# Create docker-compose.override.yml if it doesn't exist
if [[ ! -f "docker-compose.override.yml" ]]; then
    echo "📝 Creating docker-compose.override.yml..."
    cp docker-compose.override.yml.example docker-compose.override.yml
    echo "✅ Override file created - customize ports and settings as needed"
fi

# Create volumes directory
echo "📁 Setting up data volumes..."
mkdir -p volumes/uploads volumes/slideshows volumes/db volumes/redis volumes/temp

# Set proper permissions
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # On Linux, set ownership to the user that will run in the container
    sudo chown -R 1000:1000 volumes/ 2>/dev/null || true
fi

echo "🔧 Building Docker images..."
if ! docker-compose build; then
    echo "❌ Docker build failed. Check the error messages above."
    exit 1
fi

echo "🚀 Starting services..."
if ! docker-compose up -d; then
    echo "❌ Failed to start services. Check the error messages above."
    exit 1
fi

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check service health
echo "🔍 Checking service status..."
docker-compose ps

# Get the web service port
WEB_PORT=$(docker-compose port web 5000 2>/dev/null | cut -d: -f2 || echo "5000")

# Test if web service is responding
if curl -f http://localhost:${WEB_PORT}/health > /dev/null 2>&1; then
    echo
    echo "🎉 Photo Frame Slideshow Server is running!"
    echo "==========================================="
    echo
    echo "🌐 Admin Interface: http://localhost:${WEB_PORT}"
    echo "🔑 Admin Password:  changeme123 (change this in .env!)"
    echo
    echo "📖 Quick Start:"
    echo "1. Open http://localhost:${WEB_PORT} in your browser"
    echo "2. Login with the admin password"
    echo "3. Click 'Create New Slideshow'"
    echo "4. Upload a ZIP file with your photos"
    echo "5. Configure settings (resolution, weather, etc.)"
    echo "6. Wait for processing to complete"
    echo "7. Generate secure URLs to share your slideshow"
    echo
    echo "📊 Monitoring:"
    echo "   View logs:    docker-compose logs -f"
    echo "   Stop server:  docker-compose down"
    echo "   Restart:      docker-compose restart"
    echo
    echo "💾 Data Location: ./volumes/ (backup this folder!)"
    echo
else
    echo "⚠️  Services started but web interface may not be ready yet."
    echo "   Give it a few more seconds, then try: http://localhost:${WEB_PORT}"
    echo
    echo "📋 Service Status:"
    docker-compose ps
    echo
    echo "📊 Check logs if there are issues:"
    echo "   docker-compose logs web"
    echo "   docker-compose logs worker"
    echo "   docker-compose logs redis"
fi

echo "✨ Server startup complete!"