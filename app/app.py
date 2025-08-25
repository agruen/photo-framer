import os
import uuid
import secrets
import string
from datetime import datetime, timezone
from functools import wraps

from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, session, send_file, abort
from flask_socketio import SocketIO, emit, join_room, leave_room
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix

from .config import Config
from .models import db, Slideshow, SlideshowURL, ProcessingTask
from .celery_app import make_celery

# Global variables for SocketIO
socketio = None
celery = None


def create_app(config_class=Config):
    """Flask application factory"""
    global socketio, celery
    
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Add reverse proxy support
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
    
    # Initialize extensions
    db.init_app(app)
    celery = make_celery(app)
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
    
    # Create database tables
    with app.app_context():
        db.create_all()
    
    # Ensure required directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['SLIDESHOW_FOLDER'], exist_ok=True)
    os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)
    
    # Authentication decorator
    def admin_required(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not session.get('admin_logged_in'):
                return redirect(url_for('login'))
            return f(*args, **kwargs)
        return decorated_function
    
    # Routes
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            password = request.form.get('password')
            if password == app.config['ADMIN_PASSWORD']:
                session['admin_logged_in'] = True
                session.permanent = True
                flash('Successfully logged in!', 'success')
                return redirect(url_for('admin_dashboard'))
            else:
                flash('Invalid password', 'error')
        return render_template('login.html')
    
    @app.route('/logout')
    @admin_required
    def logout():
        session.clear()
        flash('Successfully logged out', 'success')
        return redirect(url_for('login'))
    
    @app.route('/')
    @admin_required
    def admin_dashboard():
        """Admin dashboard showing all slideshows"""
        slideshows = Slideshow.query.order_by(Slideshow.created_at.desc()).all()
        return render_template('dashboard.html', slideshows=slideshows)
    
    @app.route('/create', methods=['GET', 'POST'])
    @admin_required
    def create_slideshow():
        """Create new slideshow"""
        if request.method == 'POST':
            # Get form data
            name = request.form.get('name', '').strip()
            screen_width = int(request.form.get('screen_width', app.config['DEFAULT_SCREEN_WIDTH']))
            screen_height = int(request.form.get('screen_height', app.config['DEFAULT_SCREEN_HEIGHT']))
            rotation_interval = int(request.form.get('rotation_interval', app.config['DEFAULT_ROTATION_INTERVAL']))
            weather_zip = request.form.get('weather_zip', '').strip()
            weather_api_key = request.form.get('weather_api_key', '').strip()
            
            # Validation
            if not name:
                flash('Slideshow name is required', 'error')
                return render_template('create.html')
            
            if 'zip_file' not in request.files:
                flash('No zip file uploaded', 'error')
                return render_template('create.html')
            
            file = request.files['zip_file']
            if file.filename == '':
                flash('No file selected', 'error')
                return render_template('create.html')
            
            if not file.filename.lower().endswith('.zip'):
                flash('Only ZIP files are allowed', 'error')
                return render_template('create.html')
            
            # Generate unique folder name
            folder_name = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            # Save uploaded file
            filename = secure_filename(file.filename)
            unique_filename = f"{folder_name}_{filename}"
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            try:
                file.save(upload_path)
                file_size = os.path.getsize(upload_path)
                
                # Create slideshow record
                slideshow = Slideshow(
                    name=name,
                    folder_name=folder_name,
                    screen_width=screen_width,
                    screen_height=screen_height,
                    rotation_interval=rotation_interval,
                    weather_zip=weather_zip,
                    weather_api_key=weather_api_key,
                    zip_filename=unique_filename,
                    zip_size=file_size,
                    status='created'
                )
                
                db.session.add(slideshow)
                db.session.flush()  # Get the ID
                
                # Start background processing
                from .tasks import process_slideshow
                task = process_slideshow.delay(slideshow.id)
                
                # Record processing task
                processing_task = ProcessingTask(
                    slideshow_id=slideshow.id,
                    task_id=task.id
                )
                db.session.add(processing_task)
                db.session.commit()
                
                flash(f'Slideshow "{name}" created! Processing started in background.', 'success')
                return redirect(url_for('slideshow_detail', slideshow_id=slideshow.id))
                
            except Exception as e:
                # Clean up on error
                if os.path.exists(upload_path):
                    os.remove(upload_path)
                flash(f'Error creating slideshow: {str(e)}', 'error')
                return render_template('create.html')
        
        return render_template('create.html')
    
    @app.route('/slideshow/<int:slideshow_id>')
    @admin_required
    def slideshow_detail(slideshow_id):
        """Slideshow detail page with processing status"""
        slideshow = Slideshow.query.get_or_404(slideshow_id)
        return render_template('slideshow_detail.html', slideshow=slideshow)
    
    @app.route('/slideshow/<int:slideshow_id>/generate_url', methods=['POST'])
    @admin_required
    def generate_url(slideshow_id):
        """Generate a new access URL for a slideshow"""
        slideshow = Slideshow.query.get_or_404(slideshow_id)
        
        if slideshow.status != 'completed':
            flash('Cannot generate URL for incomplete slideshow', 'error')
            return redirect(url_for('slideshow_detail', slideshow_id=slideshow_id))
        
        # Generate unique URL
        url_key = SlideshowURL.generate_unique_key()
        url_name = request.form.get('url_name', '').strip()
        
        slideshow_url = SlideshowURL(
            slideshow_id=slideshow_id,
            url_key=url_key,
            name=url_name if url_name else None
        )
        
        db.session.add(slideshow_url)
        db.session.commit()
        
        flash('New access URL generated!', 'success')
        return redirect(url_for('slideshow_detail', slideshow_id=slideshow_id))
    
    @app.route('/slideshow/<int:slideshow_id>/delete', methods=['POST'])
    @admin_required
    def delete_slideshow(slideshow_id):
        """Delete a slideshow and its files"""
        slideshow = Slideshow.query.get_or_404(slideshow_id)
        
        # Start background deletion task
        from .tasks import delete_slideshow_files
        delete_slideshow_files.delay(slideshow_id)
        
        # Delete database records
        db.session.delete(slideshow)
        db.session.commit()
        
        flash(f'Slideshow "{slideshow.name}" deleted!', 'success')
        return redirect(url_for('admin_dashboard'))
    
    @app.route('/s/<url_key>')
    def view_slideshow(url_key):
        """Public slideshow viewing endpoint"""
        slideshow_url = SlideshowURL.query.filter_by(url_key=url_key).first_or_404()
        slideshow = slideshow_url.slideshow
        
        if slideshow.status != 'completed':
            abort(404)  # Hide incomplete slideshows
        
        # Record access
        slideshow_url.record_access()
        
        # Read the original slideshow HTML file
        html_path = os.path.join(app.config['SLIDESHOW_FOLDER'], slideshow.folder_name, 'slideshow.html')
        if not os.path.exists(html_path):
            abort(404)
        
        # Read and modify the HTML to include correct image paths
        with open(html_path, 'r') as f:
            html_content = f.read()
        
        # Find the image array and replace relative paths with full URLs
        import re
        import json
        
        # Extract the current images array
        images_match = re.search(r'var images = (\[.*?\]);', html_content)
        if images_match:
            current_images = json.loads(images_match.group(1))
            # Convert to full URLs with the slideshow key
            full_url_images = [f"/s/{url_key}/{img}" for img in current_images]
            # Replace in the HTML
            new_images_js = f'var images = {json.dumps(full_url_images)};'
            html_content = html_content.replace(images_match.group(0), new_images_js)
        
        return html_content, 200, {'Content-Type': 'text/html'}
    
    @app.route('/s/<url_key>/<path:filename>')
    def slideshow_static(url_key, filename):
        """Serve static files for slideshows (images, etc.)"""
        slideshow_url = SlideshowURL.query.filter_by(url_key=url_key).first_or_404()
        slideshow = slideshow_url.slideshow
        
        if slideshow.status != 'completed':
            abort(404)
        
        # Serve file from slideshow directory
        file_path = os.path.join(app.config['SLIDESHOW_FOLDER'], slideshow.folder_name, filename)
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            abort(404)
        
        return send_file(file_path)
    
    @app.route('/api/slideshow/<int:slideshow_id>/status')
    @admin_required
    def slideshow_status(slideshow_id):
        """API endpoint to get slideshow status"""
        slideshow = Slideshow.query.get_or_404(slideshow_id)
        return jsonify(slideshow.to_dict())
    
    @app.route('/health')
    def health_check():
        """Health check endpoint"""
        return jsonify({'status': 'healthy', 'timestamp': datetime.now(timezone.utc).isoformat()})
    
    # WebSocket events
    @socketio.on('connect')
    def on_connect():
        emit('connected', {'message': 'Connected to server'})
    
    @socketio.on('join_slideshow')
    def on_join_slideshow(data):
        slideshow_id = data.get('slideshow_id')
        if slideshow_id and session.get('admin_logged_in'):
            room = f'slideshow_{slideshow_id}'
            join_room(room)
            emit('joined_room', {'room': room})
    
    @socketio.on('leave_slideshow')
    def on_leave_slideshow(data):
        slideshow_id = data.get('slideshow_id')
        if slideshow_id:
            room = f'slideshow_{slideshow_id}'
            leave_room(room)
            emit('left_room', {'room': room})
    
    return app


def main():
    """Main entry point for running the app"""
    app = create_app()
    # Run with extended timeout for large uploads
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, 
                allow_unsafe_werkzeug=True, 
                ping_timeout=1800,  # 30 minutes
                ping_interval=25)


if __name__ == '__main__':
    main()