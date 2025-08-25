from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone
import secrets
import string
import os

db = SQLAlchemy()

class Slideshow(db.Model):
    """Model for slideshow metadata"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    folder_name = db.Column(db.String(200), unique=True, nullable=False)
    
    # Processing settings
    screen_width = db.Column(db.Integer, nullable=False, default=1280)
    screen_height = db.Column(db.Integer, nullable=False, default=800) 
    rotation_interval = db.Column(db.Integer, nullable=False, default=60)
    
    # Weather settings
    weather_zip = db.Column(db.String(20))
    weather_api_key = db.Column(db.String(100))
    
    # Status tracking
    status = db.Column(db.String(50), nullable=False, default='created')  # created, processing, completed, error
    progress = db.Column(db.Integer, default=0)  # 0-100
    error_message = db.Column(db.Text)
    
    # Metadata
    total_images = db.Column(db.Integer, default=0)
    processed_images = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = db.Column(db.DateTime)
    
    # File info
    zip_filename = db.Column(db.String(200))
    zip_size = db.Column(db.BigInteger)  # Size in bytes
    
    # Relationships
    urls = db.relationship('SlideshowURL', backref='slideshow', lazy=True, cascade='all, delete-orphan')
    processing_tasks = db.relationship('ProcessingTask', backref='slideshow', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Slideshow {self.name}>'
    
    @property
    def slideshow_path(self):
        """Get the full path to the slideshow folder"""
        from .config import Config
        return os.path.join(Config.SLIDESHOW_FOLDER, self.folder_name)
    
    @property
    def html_path(self):
        """Get the path to the slideshow HTML file"""
        return os.path.join(self.slideshow_path, 'slideshow.html')
    
    def to_dict(self):
        """Convert slideshow to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'folder_name': self.folder_name,
            'screen_width': self.screen_width,
            'screen_height': self.screen_height,
            'rotation_interval': self.rotation_interval,
            'weather_zip': self.weather_zip,
            'status': self.status,
            'progress': self.progress,
            'error_message': self.error_message,
            'total_images': self.total_images,
            'processed_images': self.processed_images,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'zip_filename': self.zip_filename,
            'zip_size': self.zip_size,
            'url_count': len(self.urls)
        }


class SlideshowURL(db.Model):
    """Model for slideshow access URLs with 256-character random keys"""
    id = db.Column(db.Integer, primary_key=True)
    slideshow_id = db.Column(db.Integer, db.ForeignKey('slideshow.id'), nullable=False)
    url_key = db.Column(db.String(256), unique=True, nullable=False)
    name = db.Column(db.String(200))  # Optional name for the URL
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    access_count = db.Column(db.Integer, default=0)
    last_accessed = db.Column(db.DateTime)
    
    def __repr__(self):
        return f'<SlideshowURL {self.url_key[:20]}...>'
    
    @staticmethod
    def generate_unique_key():
        """Generate a unique 256-character random key"""
        # Use URL-safe characters (letters, numbers, - and _)
        alphabet = string.ascii_letters + string.digits + '-_'
        while True:
            key = ''.join(secrets.choice(alphabet) for _ in range(256))
            # Check if key already exists
            if not SlideshowURL.query.filter_by(url_key=key).first():
                return key
    
    def record_access(self):
        """Record an access to this URL"""
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc)
        db.session.commit()
    
    def to_dict(self):
        """Convert URL to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'slideshow_id': self.slideshow_id,
            'url_key': self.url_key,
            'name': self.name,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None
        }


class ProcessingTask(db.Model):
    """Model for tracking processing tasks (Celery task IDs)"""
    id = db.Column(db.Integer, primary_key=True)
    slideshow_id = db.Column(db.Integer, db.ForeignKey('slideshow.id'), nullable=False)
    task_id = db.Column(db.String(200), unique=True, nullable=False)
    task_type = db.Column(db.String(50), nullable=False, default='process_slideshow')
    status = db.Column(db.String(50), nullable=False, default='pending')
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    started_at = db.Column(db.DateTime)
    completed_at = db.Column(db.DateTime)
    
    
    def __repr__(self):
        return f'<ProcessingTask {self.task_id}>'
    
    def to_dict(self):
        """Convert task to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'slideshow_id': self.slideshow_id,
            'task_id': self.task_id,
            'task_type': self.task_type,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }