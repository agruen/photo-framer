from celery import Celery
from .config import Config

def make_celery(app):
    """Create Celery instance and configure it with Flask app"""
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    
    # Update configuration from Flask config
    celery.conf.update(
        task_serializer=app.config['CELERY_TASK_SERIALIZER'],
        accept_content=app.config['CELERY_ACCEPT_CONTENT'],
        result_serializer=app.config['CELERY_RESULT_SERIALIZER'],
        timezone=app.config['CELERY_TIMEZONE'],
        enable_utc=app.config['CELERY_ENABLE_UTC'],
        task_track_started=True,
        task_routes={
            'app.tasks.process_slideshow': {'queue': 'processing'},
        }
    )
    
    # Create task context manager that preserves Flask app context
    class ContextTask(celery.Task):
        """Make celery tasks work with Flask app context"""
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)
    
    celery.Task = ContextTask
    return celery


# Standalone Celery app for worker processes
def create_celery_app():
    """Create standalone Celery app for worker processes"""
    celery = Celery('slideshow_processor')
    
    # Load configuration from environment/config
    celery.conf.update(
        broker_url=Config.CELERY_BROKER_URL,
        result_backend=Config.CELERY_RESULT_BACKEND,
        task_serializer=Config.CELERY_TASK_SERIALIZER,
        accept_content=Config.CELERY_ACCEPT_CONTENT,
        result_serializer=Config.CELERY_RESULT_SERIALIZER,
        timezone=Config.CELERY_TIMEZONE,
        enable_utc=Config.CELERY_ENABLE_UTC,
        task_track_started=True,
        task_routes={
            'app.tasks.process_slideshow': {'queue': 'processing'},
        }
    )
    
    return celery