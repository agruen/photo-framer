# Database Migrations

This directory contains database migration scripts for Photo-Framer.

## Running Migrations

### For Docker Deployments

Migrations will run automatically when the container starts. The startup script checks for and applies any pending migrations.

### Manual Migration

If you need to run migrations manually:

```bash
# Inside the Docker container
docker-compose exec web python3 migrations/add_per_link_weather.py migrate

# Or locally (if running outside Docker)
python3 migrations/add_per_link_weather.py migrate
```

## Available Migrations

### add_per_link_weather.py

**Purpose:** Adds per-link weather configuration to SlideshowURL model

**Changes:**
- Adds `weather_zip` column to `slideshow_url` table
- Adds `weather_api_key` column to `slideshow_url` table

**Usage:**
```bash
# Apply migration
python3 migrations/add_per_link_weather.py migrate

# Check status (shows if already applied)
python3 migrations/add_per_link_weather.py migrate
```

**Note:** SQLite does not support DROP COLUMN, so rollback is not supported. Back up your database before migrating.

## Migration Workflow

1. Code changes are made to models in `app/models.py`
2. Migration script is created in `migrations/` directory
3. Migration is tested locally
4. Migration is committed to version control
5. On deployment, migration runs automatically (or manually via script)

## Database Location

- **Docker:** `/app/db/slideshows.db` (persisted in `volumes/db/`)
- **Local:** Configured via `SQLALCHEMY_DATABASE_URI` in `app/config.py`
