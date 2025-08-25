#!/usr/bin/env python3
"""
Database migration script to handle the relationship changes.
This will recreate the database with the proper cascade relationships.
"""
import os
import sys
import shutil
from datetime import datetime

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.app import create_app
from app.models import db, Slideshow, SlideshowURL, ProcessingTask

def migrate_database():
    """Recreate the database with proper relationships"""
    app = create_app()
    
    with app.app_context():
        # Backup existing data if database exists
        backup_data = {}
        
        try:
            # Try to read existing data
            slideshows = Slideshow.query.all()
            urls = SlideshowURL.query.all()
            
            backup_data['slideshows'] = [s.to_dict() for s in slideshows]
            backup_data['urls'] = [u.to_dict() for u in urls]
            
            print(f"Backed up {len(backup_data['slideshows'])} slideshows and {len(backup_data['urls'])} URLs")
            
        except Exception as e:
            print(f"No existing data to backup: {e}")
            backup_data = {'slideshows': [], 'urls': []}
        
        # Drop and recreate all tables
        print("Dropping existing tables...")
        db.drop_all()
        
        print("Creating new tables with proper relationships...")
        db.create_all()
        
        # Restore data (excluding processing_tasks as they are likely stale)
        if backup_data['slideshows']:
            print("Restoring slideshow data...")
            for slideshow_data in backup_data['slideshows']:
                # Remove fields that shouldn't be restored
                slideshow_data.pop('id', None)
                slideshow_data.pop('created_at', None)
                slideshow_data.pop('completed_at', None)
                slideshow_data.pop('url_count', None)
                
                # Convert ISO strings back to datetime objects
                if slideshow_data.get('created_at'):
                    slideshow_data['created_at'] = datetime.fromisoformat(slideshow_data['created_at'].replace('Z', '+00:00'))
                if slideshow_data.get('completed_at'):
                    slideshow_data['completed_at'] = datetime.fromisoformat(slideshow_data['completed_at'].replace('Z', '+00:00'))
                
                slideshow = Slideshow(**slideshow_data)
                db.session.add(slideshow)
            
            db.session.flush()  # Get IDs
            
            # Restore URLs (need to match them with new slideshow IDs)
            print("Restoring URL data...")
            slideshow_mapping = {}
            for i, slideshow in enumerate(db.session.query(Slideshow).all()):
                slideshow_mapping[backup_data['slideshows'][i]['folder_name']] = slideshow.id
            
            for url_data in backup_data['urls']:
                # Find the corresponding slideshow by folder_name
                original_slideshow = next((s for s in backup_data['slideshows'] if s['id'] == url_data['slideshow_id']), None)
                if original_slideshow:
                    new_slideshow_id = slideshow_mapping.get(original_slideshow['folder_name'])
                    if new_slideshow_id:
                        url_data.pop('id', None)
                        url_data['slideshow_id'] = new_slideshow_id
                        
                        if url_data.get('created_at'):
                            url_data['created_at'] = datetime.fromisoformat(url_data['created_at'].replace('Z', '+00:00'))
                        if url_data.get('last_accessed'):
                            url_data['last_accessed'] = datetime.fromisoformat(url_data['last_accessed'].replace('Z', '+00:00'))
                        
                        url = SlideshowURL(**url_data)
                        db.session.add(url)
            
            db.session.commit()
            print("Data restoration complete!")
        
        print("Database migration completed successfully!")

if __name__ == '__main__':
    migrate_database()