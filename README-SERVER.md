# Photo Frame Slideshow Server

A complete Dockerized web application for creating and hosting photo slideshows for digital photo frames. Upload ZIP files of photos, process them with face-aware cropping, and create beautiful slideshows with weather integration.

## Features

### 🖼️ Photo Processing
- **Face-aware cropping** with advanced detection algorithms
- **Batch processing** with multi-core support
- **Multiple format support** (JPG, PNG, BMP, TIFF)
- **Automatic image optimization** and resizing
- **Background processing** with real-time progress updates

### 🌐 Web Interface
- **Drag-and-drop upload** for large ZIP files (up to 10GB)
- **Real-time processing status** with WebSocket updates
- **Secure admin authentication**
- **Mobile-responsive design**
- **Progress tracking** and error handling

### 📱 Slideshow Features
- **Multiple protected URLs** per slideshow (256-char random keys)
- **Weather integration** with OpenWeatherMap API
- **Real-time clock display**
- **Configurable image rotation** (5-600 seconds)
- **Custom screen resolutions**
- **Full-screen optimized** for any display device

### 🐳 Infrastructure
- **Docker Compose** setup with persistent volumes
- **Multi-architecture support** (x86_64, ARM64 for Raspberry Pi)
- **Redis background processing** with Celery
- **SQLite database** with automatic setup
- **Health checks** and container monitoring

## Quick Start

### 1. Clone and Setup
```bash
git clone <your-repo>
cd Photo-Framer

# Copy environment configuration
cp .env.example .env
cp docker-compose.override.yml.example docker-compose.override.yml

# Edit configuration (optional)
nano .env
nano docker-compose.override.yml
```

### 2. Start the Server
```bash
# Build and start all services
docker-compose up -d --build

# Check status
docker-compose ps

# View logs
docker-compose logs -f web
```

### 3. Access Admin Interface
1. Open http://localhost:5000 in your browser
2. Login with password: `changeme123` (change this in .env file!)
3. Click "Create New Slideshow"

### 4. Create Your First Slideshow
1. **Upload**: Drag and drop a ZIP file containing your photos
2. **Configure**: Set screen resolution, weather location, API key
3. **Process**: Watch real-time progress as images are processed
4. **Share**: Generate secure URLs to share your slideshow

## Configuration

### Environment Variables (.env)
```env
# Security
ADMIN_PASSWORD=your-secure-password
SECRET_KEY=your-random-secret-key

# File Upload
MAX_CONTENT_LENGTH=10737418240  # 10GB max

# Default Settings
DEFAULT_SCREEN_WIDTH=1280
DEFAULT_SCREEN_HEIGHT=800
DEFAULT_ROTATION_INTERVAL=60
```

### Docker Compose Override
Customize ports, resource limits, and other Docker settings:
```yaml
version: '3.8'
services:
  web:
    ports:
      - "8080:5000"  # Change external port
    environment:
      - ADMIN_PASSWORD=your-secure-password
```

## Weather Integration

1. Get a free API key from [OpenWeatherMap](https://openweathermap.org/api)
2. When creating a slideshow, enter:
   - **ZIP Code**: Your location (US format, e.g., "10001")
   - **API Key**: Your OpenWeatherMap API key

Weather data includes temperature and weather icons, refreshed every 5 minutes.

## Volume Structure

All data is stored in `./volumes/` directory:
```
volumes/
├── uploads/     # Temporary ZIP file storage (auto-cleaned)
├── slideshows/  # Processed slideshows and HTML files
├── db/          # SQLite database
├── redis/       # Redis data
└── temp/        # Processing temporary files
```

**Backup**: Simply backup the entire `volumes/` folder to preserve all your slideshows and data.

## API Endpoints

### Admin Endpoints (require authentication)
- `GET /` - Admin dashboard
- `GET /create` - Create slideshow form
- `POST /create` - Upload and process slideshow
- `GET /slideshow/<id>` - Slideshow detail page
- `POST /slideshow/<id>/generate_url` - Generate new access URL
- `DELETE /slideshow/<id>/delete` - Delete slideshow

### Public Endpoints
- `GET /s/<256-char-key>` - View slideshow (no auth required)
- `GET /s/<256-char-key>/<filename>` - Serve slideshow assets

### Health Check
- `GET /health` - Service health status

## Security Features

- **Admin authentication** with session management
- **256-character random URLs** for slideshow access
- **File type validation** (ZIP files only)
- **Path sanitization** to prevent directory traversal
- **Secure file handling** with temporary storage cleanup
- **Rate limiting** and upload size restrictions

## Troubleshooting

### Common Issues

**Build fails on Raspberry Pi:**
```bash
# Use ARM-specific build
docker-compose build --build-arg TARGETPLATFORM=linux/arm64
```

**Upload fails:**
```bash
# Check disk space
df -h

# Check Docker logs
docker-compose logs web worker
```

**Processing stuck:**
```bash
# Restart worker
docker-compose restart worker

# Check Redis connection
docker-compose exec redis redis-cli ping
```

**Permission errors:**
```bash
# Fix volume permissions
sudo chown -R 1000:1000 volumes/
```

### Performance Tuning

**For large photo collections:**
```yaml
# In docker-compose.override.yml
services:
  worker:
    environment:
      - CELERYD_CONCURRENCY=4  # Increase workers
    deploy:
      resources:
        limits:
          memory: 4G  # Increase memory limit
```

**For Raspberry Pi:**
```yaml
services:
  worker:
    environment:
      - CELERYD_CONCURRENCY=2  # Reduce workers
    deploy:
      resources:
        limits:
          memory: 1G  # Limit memory usage
```

## Development

### Running Locally (without Docker)
```bash
# Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-server.txt

# Setup Redis
brew install redis  # macOS
sudo apt install redis-server  # Ubuntu

# Start services
redis-server &
celery -A app.tasks worker --loglevel=info &
python -m app.app
```

### Project Structure
```
Photo-Framer/
├── app/
│   ├── __init__.py
│   ├── app.py           # Flask application
│   ├── models.py        # Database models
│   ├── tasks.py         # Celery background tasks
│   ├── config.py        # Configuration
│   ├── celery_app.py    # Celery setup
│   └── templates/       # HTML templates
├── face_crop_tool.py    # Face detection & cropping
├── slideshow_generator.py  # HTML slideshow generation
├── docker-compose.yml   # Docker services
├── Dockerfile          # Container definition
└── volumes/            # Persistent data
```

## License

This project combines face-aware photo cropping with web-based slideshow hosting. Built with Flask, Celery, Redis, and Docker.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review Docker logs: `docker-compose logs`
3. Open an issue on the repository