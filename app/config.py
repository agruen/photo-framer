import os
from datetime import timedelta

class Config:
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
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
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or 'uploads'
    SLIDESHOW_FOLDER = os.environ.get('SLIDESHOW_FOLDER') or 'slideshows'
    TEMP_FOLDER = os.environ.get('TEMP_FOLDER') or 'temp'
    
    # Admin configuration
    ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD') or 'admin123'
    
    # Session configuration
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # File upload configuration
    ALLOWED_EXTENSIONS = {'zip'}
    CHUNK_SIZE = 1024 * 1024  # 1MB chunks for upload
    
    # Processing configuration
    DEFAULT_SCREEN_WIDTH = 1280
    DEFAULT_SCREEN_HEIGHT = 800
    DEFAULT_ROTATION_INTERVAL = 60  # seconds
    URL_KEY_LENGTH = 256
    
    @staticmethod
    def init_app(app):
        pass