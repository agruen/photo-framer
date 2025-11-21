import eventlet
eventlet.monkey_patch()

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

    # Initialize config from setup
    config_class.init_app(app)

    # Add reverse proxy support
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

    # Initialize extensions
    db.init_app(app)
    celery = make_celery(app)
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', message_queue=app.config['CELERY_BROKER_URL'])

    # Create database tables
    with app.app_context():
        db.create_all()

    # Ensure required directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['SLIDESHOW_FOLDER'], exist_ok=True)
    os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)

    # Setup middleware - redirect to setup if not complete
    @app.before_request
    def check_setup():
        from .setup import is_setup_complete
        # Skip setup check for setup routes, static files, and health check
        if request.endpoint and (
            request.endpoint.startswith('setup') or
            request.endpoint == 'static' or
            request.endpoint == 'health_check'
        ):
            return None

        if not is_setup_complete():
            return redirect(url_for('setup_wizard'))

    # Authentication decorator
    def admin_required(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not session.get('admin_logged_in'):
                return redirect(url_for('login'))
            return f(*args, **kwargs)
        return decorated_function

    # Setup Routes
    @app.route('/setup', methods=['GET', 'POST'])
    def setup_wizard():
        """First-time setup wizard"""
        from .setup import is_setup_complete, save_config, mark_setup_complete

        # Redirect to dashboard if already set up
        if is_setup_complete():
            return redirect(url_for('admin_dashboard'))

        if request.method == 'POST':
            try:
                # Get form data
                admin_password = request.form.get('admin_password', '').strip()
                admin_password_confirm = request.form.get('admin_password_confirm', '').strip()
                screen_width = int(request.form.get('screen_width', 1280))
                screen_height = int(request.form.get('screen_height', 800))
                rotation_interval = int(request.form.get('rotation_interval', 60))

                # Validation
                if not admin_password:
                    flash('Admin password is required', 'error')
                    return render_template('setup.html')

                if len(admin_password) < 8:
                    flash('Admin password must be at least 8 characters', 'error')
                    return render_template('setup.html')

                if admin_password != admin_password_confirm:
                    flash('Passwords do not match', 'error')
                    return render_template('setup.html')

                # Save configuration
                config_data = {
                    'admin_password': admin_password,
                    'default_screen_width': screen_width,
                    'default_screen_height': screen_height,
                    'default_rotation_interval': rotation_interval
                }
                save_config(config_data)

                # Mark setup as complete
                mark_setup_complete()

                # Update app config with new values
                app.config['ADMIN_PASSWORD'] = admin_password
                app.config['DEFAULT_SCREEN_WIDTH'] = screen_width
                app.config['DEFAULT_SCREEN_HEIGHT'] = screen_height
                app.config['DEFAULT_ROTATION_INTERVAL'] = rotation_interval

                flash('Setup completed successfully! Please log in with your admin password.', 'success')
                return redirect(url_for('login'))

            except ValueError as e:
                flash(f'Invalid input: {str(e)}', 'error')
                return render_template('setup.html')
            except Exception as e:
                flash(f'Setup error: {str(e)}', 'error')
                return render_template('setup.html')

        return render_template('setup.html')

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
                # Stream large files to disk to avoid memory issues
                file_size = 0
                with open(upload_path, 'wb') as f:
                    while True:
                        chunk = file.stream.read(app.config.get('CHUNK_SIZE', 1024*1024))  # 1MB chunks
                        if not chunk:
                            break
                        f.write(chunk)
                        file_size += len(chunk)
                
                # Verify file was written correctly
                if file_size == 0:
                    raise Exception("File upload failed - no data received")
                
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
    
    @app.route('/create_folder_slideshow', methods=['POST'])
    @admin_required
    def create_folder_slideshow():
        """Create slideshow record for folder uploads"""
        try:
            # Get form data
            name = request.form.get('name', '').strip()
            screen_width = int(request.form.get('screen_width', app.config['DEFAULT_SCREEN_WIDTH']))
            screen_height = int(request.form.get('screen_height', app.config['DEFAULT_SCREEN_HEIGHT']))
            rotation_interval = int(request.form.get('rotation_interval', app.config['DEFAULT_ROTATION_INTERVAL']))
            weather_zip = request.form.get('weather_zip', '').strip()
            weather_api_key = request.form.get('weather_api_key', '').strip()
            total_files = int(request.form.get('total_files', 0))
            
            # Validation
            if not name:
                return jsonify({'error': 'Slideshow name is required'}), 400
            
            if total_files == 0:
                return jsonify({'error': 'No files to upload'}), 400
            
            # Generate unique folder name
            folder_name = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            # Create slideshow record
            slideshow = Slideshow(
                name=name,
                folder_name=folder_name,
                screen_width=screen_width,
                screen_height=screen_height,
                rotation_interval=rotation_interval,
                weather_zip=weather_zip,
                weather_api_key=weather_api_key,
                zip_filename=None,  # No ZIP for folder uploads
                zip_size=0,
                status='uploading'  # New status for folder uploads
            )
            
            db.session.add(slideshow)
            db.session.flush()  # Get the ID
            
            # Create slideshow directory
            slideshow_dir = os.path.join(app.config['SLIDESHOW_FOLDER'], folder_name)
            os.makedirs(slideshow_dir, exist_ok=True)
            
            db.session.commit()
            
            return jsonify({'slideshow_id': slideshow.id, 'folder_name': folder_name})
            
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': str(e)}), 500
    
    @app.route('/upload_batch', methods=['POST'])
    @admin_required
    def upload_batch():
        """Handle batch file uploads for folder slideshows"""
        try:
            slideshow_id = request.form.get('slideshow_id')
            batch_index = request.form.get('batch_index')
            
            slideshow = Slideshow.query.get_or_404(slideshow_id)
            
            if slideshow.status != 'uploading':
                return jsonify({'error': 'Slideshow not in upload state'}), 400
            
            slideshow_dir = os.path.join(app.config['SLIDESHOW_FOLDER'], slideshow.folder_name)
            
            # Process uploaded files
            files = request.files.getlist('files[]')
            file_paths = request.form.getlist('file_paths[]')
            
            if not files or len(files) == 0:
                return jsonify({'error': 'No files in batch'}), 400
            
            saved_files = []
            for i, file in enumerate(files):
                if file and file.filename:
                    # Get original path if available, otherwise use filename
                    original_path = file_paths[i] if i < len(file_paths) else file.filename
                    
                    # Create subdirectories if needed
                    if '/' in original_path:
                        relative_dir = os.path.dirname(original_path)
                        full_dir = os.path.join(slideshow_dir, relative_dir)
                        os.makedirs(full_dir, exist_ok=True)
                    
                    # Save file with original name/path structure
                    safe_filename = secure_filename(os.path.basename(original_path))
                    if '/' in original_path:
                        relative_dir = os.path.dirname(original_path)
                        file_path = os.path.join(slideshow_dir, relative_dir, safe_filename)
                    else:
                        file_path = os.path.join(slideshow_dir, safe_filename)
                    
                    file.save(file_path)
                    saved_files.append(original_path)
            
            return jsonify({
                'success': True, 
                'batch_index': batch_index,
                'files_saved': len(saved_files),
                'files': saved_files
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/finalize_slideshow', methods=['POST'])
    @admin_required
    def finalize_slideshow():
        """Finalize slideshow after all files are uploaded"""
        try:
            data = request.get_json()
            slideshow_id = data.get('slideshow_id')
            
            slideshow = Slideshow.query.get_or_404(slideshow_id)
            
            if slideshow.status != 'uploading':
                return jsonify({'error': 'Slideshow not in upload state'}), 400
            
            # Update status to processing
            slideshow.status = 'processing'
            db.session.commit()
            
            # Start background processing
            from .tasks import process_folder_slideshow
            task = process_folder_slideshow.delay(slideshow.id)
            
            # Record processing task
            processing_task = ProcessingTask(
                slideshow_id=slideshow.id,
                task_id=task.id
            )
            db.session.add(processing_task)
            db.session.commit()
            
            return jsonify({'success': True, 'task_id': task.id})
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
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

        # Get per-link weather configuration (optional overrides)
        weather_zip = request.form.get('weather_zip', '').strip()
        weather_api_key = request.form.get('weather_api_key', '').strip()

        slideshow_url = SlideshowURL(
            slideshow_id=slideshow_id,
            url_key=url_key,
            name=url_name if url_name else None,
            weather_zip=weather_zip if weather_zip else None,
            weather_api_key=weather_api_key if weather_api_key else None
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

    @app.route('/slideshow/<int:slideshow_id>/add_photos', methods=['GET'])
    @admin_required
    def add_photos(slideshow_id):
        """Page to add more photos to an existing slideshow"""
        slideshow = Slideshow.query.get_or_404(slideshow_id)

        if slideshow.status != 'completed':
            flash('Can only add photos to completed slideshows', 'error')
            return redirect(url_for('slideshow_detail', slideshow_id=slideshow_id))

        return render_template('add_photos.html', slideshow=slideshow)

    @app.route('/slideshow/<int:slideshow_id>/upload_batch_addition', methods=['POST'])
    @admin_required
    def upload_batch_addition(slideshow_id):
        """Handle batch file uploads for adding photos to existing slideshow"""
        try:
            batch_index = request.form.get('batch_index')

            slideshow = Slideshow.query.get_or_404(slideshow_id)

            # Only allow adding photos to completed slideshows
            if slideshow.status not in ['completed', 'uploading']:
                return jsonify({'error': 'Can only add photos to completed slideshows'}), 400

            # Update status to uploading if it was completed
            if slideshow.status == 'completed':
                slideshow.status = 'uploading'
                db.session.commit()

            slideshow_dir = os.path.join(app.config['SLIDESHOW_FOLDER'], slideshow.folder_name)

            # Process uploaded files
            files = request.files.getlist('files[]')
            file_paths = request.form.getlist('file_paths[]')

            if not files or len(files) == 0:
                return jsonify({'error': 'No files in batch'}), 400

            saved_files = []
            for i, file in enumerate(files):
                if file and file.filename:
                    # Get original filename
                    original_filename = file_paths[i] if i < len(file_paths) else file.filename

                    # Save file with original name (will be processed and renamed later)
                    safe_filename = secure_filename(os.path.basename(original_filename))
                    file_path = os.path.join(slideshow_dir, safe_filename)

                    # If file already exists, add a timestamp to avoid overwriting
                    if os.path.exists(file_path):
                        name, ext = os.path.splitext(safe_filename)
                        safe_filename = f"{name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}{ext}"
                        file_path = os.path.join(slideshow_dir, safe_filename)

                    file.save(file_path)
                    saved_files.append(original_filename)

            return jsonify({
                'success': True,
                'batch_index': batch_index,
                'files_saved': len(saved_files),
                'files': saved_files
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/slideshow/<int:slideshow_id>/finalize_addition', methods=['POST'])
    @admin_required
    def finalize_addition(slideshow_id):
        """Finalize photo addition after all files are uploaded"""
        try:
            slideshow = Slideshow.query.get_or_404(slideshow_id)

            if slideshow.status != 'uploading':
                return jsonify({'error': 'Slideshow not in upload state'}), 400

            # Update status to processing
            slideshow.status = 'processing'
            db.session.commit()

            # Start background processing to add photos
            from .tasks import add_photos_to_slideshow
            task = add_photos_to_slideshow.delay(slideshow.id)

            # Record processing task
            processing_task = ProcessingTask(
                slideshow_id=slideshow.id,
                task_id=task.id,
                task_type='add_photos'
            )
            db.session.add(processing_task)
            db.session.commit()

            return jsonify({'success': True, 'task_id': task.id})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/slideshow/<int:slideshow_id>/download')
    @admin_required
    def download_slideshow(slideshow_id):
        """Download slideshow as ZIP file for local use"""
        slideshow = Slideshow.query.get_or_404(slideshow_id)

        if slideshow.status != 'completed':
            flash('Can only download completed slideshows', 'error')
            return redirect(url_for('slideshow_detail', slideshow_id=slideshow_id))

        import zipfile
        import tempfile
        from io import BytesIO

        try:
            # Create a BytesIO object to store the ZIP in memory
            memory_file = BytesIO()

            # Get slideshow folder path
            slideshow_folder = os.path.join(app.config['SLIDESHOW_FOLDER'], slideshow.folder_name)

            if not os.path.exists(slideshow_folder):
                flash('Slideshow folder not found', 'error')
                return redirect(url_for('slideshow_detail', slideshow_id=slideshow_id))

            # Create ZIP file
            with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Walk through the slideshow folder
                for root, dirs, files in os.walk(slideshow_folder):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Add file to ZIP with relative path
                        arcname = os.path.relpath(file_path, slideshow_folder)
                        zipf.write(file_path, arcname)

            # Seek to the beginning of the BytesIO object
            memory_file.seek(0)

            # Generate safe filename
            safe_name = "".join(c for c in slideshow.name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"{safe_name}_slideshow.zip"

            return send_file(
                memory_file,
                mimetype='application/zip',
                as_attachment=True,
                download_name=filename
            )

        except Exception as e:
            flash(f'Error creating download: {str(e)}', 'error')
            return redirect(url_for('slideshow_detail', slideshow_id=slideshow_id))

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

        # Inject per-link weather configuration if present
        if slideshow_url.weather_zip or slideshow_url.weather_api_key:
            weather_override_script = '<script>\n'
            if slideshow_url.weather_zip:
                # Override the zip code by replacing the startSlideshow call
                zip_pattern = r"startSlideshow\('([^']*)'\);"
                new_zip_call = f"startSlideshow('{slideshow_url.weather_zip}');"
                html_content = re.sub(zip_pattern, new_zip_call, html_content)

            if slideshow_url.weather_api_key:
                # Override the API key before startSlideshow is called
                weather_override_script += f"  var apiKey = '{slideshow_url.weather_api_key}';\n"

            weather_override_script += '</script>\n'

            # Inject the script just before the startSlideshow call
            if slideshow_url.weather_api_key:
                # Find the location where startSlideshow is called and inject before it
                start_pattern = r"(\s*startSlideshow\(')"
                html_content = re.sub(start_pattern, weather_override_script + r'\1', html_content, count=1)

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
            app.logger.info(f"Client joined room: {room}")
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
    
    # Run with socketio - eventlet handles large uploads well
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)


if __name__ == '__main__':
    main()