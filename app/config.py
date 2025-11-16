import os
from datetime import timedelta

class Config:
    # Flask configuration
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 10 * 1024 * 1024 * 1024))  # 10GB default

    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:////app/db/slideshows.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Celery configuration
    CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL') or 'redis://localhost:6379/0'
    CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND') or 'redis://localhost:6379/0'
    CELERY_TASK_SERIALIZER = 'json'
    CELERY_ACCEPT_CONTENT = ['json']
    CELERY_RESULT_SERIALIZER = 'json'
    CELERY_TIMEZONE = 'UTC'
    CELERY_ENABLE_UTC = True

    # Upload configuration
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or '/app/uploads'
    SLIDESHOW_FOLDER = os.environ.get('SLIDESHOW_FOLDER') or '/app/slideshows'
    TEMP_FOLDER = os.environ.get('TEMP_FOLDER') or '/app/temp'

    # Session configuration
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)

    # File upload configuration
    ALLOWED_EXTENSIONS = {'zip'}
    CHUNK_SIZE = 1024 * 1024  # 1MB chunks for upload

    # Processing configuration (defaults - can be overridden by setup)
    DEFAULT_SCREEN_WIDTH = 1280
    DEFAULT_SCREEN_HEIGHT = 800
    DEFAULT_ROTATION_INTERVAL = 60  # seconds
    URL_KEY_LENGTH = 256

    # Reverse proxy support - auto-detect from headers
    PREFERRED_URL_SCHEME = None  # Auto-detect from X-Forwarded-Proto
    SERVER_NAME = None  # Auto-detect from X-Forwarded-Host

    # These will be set by init_app from setup config
    SECRET_KEY = None
    ADMIN_PASSWORD = None

    @staticmethod
    def init_app(app):
        # Load configuration from setup
        from .setup import get_secret_key, get_admin_password, get_default_settings

        # Set secret key and admin password
        app.config['SECRET_KEY'] = get_secret_key()
        app.config['ADMIN_PASSWORD'] = get_admin_password()

        # Load default settings
        defaults = get_default_settings()
        app.config['DEFAULT_SCREEN_WIDTH'] = defaults['screen_width']
        app.config['DEFAULT_SCREEN_HEIGHT'] = defaults['screen_height']
        app.config['DEFAULT_ROTATION_INTERVAL'] = defaults['rotation_interval']