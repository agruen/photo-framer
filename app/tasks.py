import os
import shutil
import zipfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import logging

from celery import Celery
from .models import db, Slideshow, ProcessingTask
from .config import Config

# Import existing photo processing modules
import sys
sys.path.append('/app')  # Add app directory to Python path
from face_crop_tool import FaceAwareCropper
from slideshow_generator import generate_slideshow_html

# HEIC conversion support
import pyheif
from PIL import Image


def convert_heic_to_jpeg(heic_path):
    """Converts a HEIC file to JPEG and returns the new path."""
    try:
        heif_file = pyheif.read(heic_path)
        image = Image.frombytes(
            heif_file.mode, 
            heif_file.size, 
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
        )
        
        jpeg_path = os.path.splitext(heic_path)[0] + ".jpg"
        image.save(jpeg_path, "JPEG")
        
        # Remove original HEIC file to save space
        os.remove(heic_path)
        
        logger.info(f"Converted HEIC file '{heic_path}' to '{jpeg_path}'")
        return jpeg_path
    except Exception as e:
        logger.error(f"Failed to convert HEIC file '{heic_path}': {e}")
        return None



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Celery instance
celery = Celery('slideshow_processor')
celery.conf.update(
    broker_url=Config.CELERY_BROKER_URL,
    result_backend=Config.CELERY_RESULT_BACKEND,
    task_serializer=Config.CELERY_TASK_SERIALIZER,
    accept_content=Config.CELERY_ACCEPT_CONTENT,
    result_serializer=Config.CELERY_RESULT_SERIALIZER,
    timezone=Config.CELERY_TIMEZONE,
    enable_utc=Config.CELERY_ENABLE_UTC,
    task_track_started=True,
)


class DatabaseTask(celery.Task):
    """Base task that handles database operations"""
    _db = None
    
    @property
    def db_session(self):
        if self._db is None:
            from app.app import create_app
            app = create_app()
            self._db = app
        return self._db


@celery.task(bind=True, base=DatabaseTask)
def process_slideshow(self, slideshow_id):
    """
    Background task to process a slideshow from uploaded zip file.
    
    Steps:
    1. Extract zip file to temporary directory
    2. Process images with face-aware cropping
    3. Generate slideshow HTML
    4. Clean up temporary files and uploaded zip
    5. Update database with results
    """
    
    logger.info(f"Starting slideshow processing for slideshow_id: {slideshow_id}")
    
    with self.db_session.app_context():
        # Get slideshow from database
        slideshow = Slideshow.query.get(slideshow_id)
        if not slideshow:
            logger.error(f"Slideshow {slideshow_id} not found")
            return {'error': 'Slideshow not found'}
        
        try:
            # Update status to processing
            slideshow.status = 'processing'
            slideshow.progress = 0
            db.session.commit()
            
            # Emit progress update via WebSocket (if available)
            emit_progress_update(slideshow_id, 'processing', 0, 'Starting processing...')
            
            # Define paths
            upload_path = os.path.join(Config.UPLOAD_FOLDER, slideshow.zip_filename)
            output_path = os.path.join(Config.SLIDESHOW_FOLDER, slideshow.folder_name)
            temp_extract_path = os.path.join(Config.TEMP_FOLDER, f"extract_{slideshow_id}")
            
            # Create output directory
            os.makedirs(output_path, exist_ok=True)
            os.makedirs(temp_extract_path, exist_ok=True)
            
            # Step 1: Extract zip file
            logger.info(f"Extracting zip file: {upload_path}")
            emit_progress_update(slideshow_id, 'processing', 5, 'Extracting zip file...')
            
            image_files = []
            supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.heic', '.heif'}
            
            with zipfile.ZipFile(upload_path, 'r') as zip_ref:
                # Get list of image files in zip
                all_files = zip_ref.namelist()
                image_files_in_zip = [
                    f for f in all_files 
                    if any(f.lower().endswith(ext) for ext in supported_extensions)
                    and not f.startswith('__MACOSX/')  # Skip Mac metadata
                    and not f.startswith('.')  # Skip hidden files
                ]
                
                logger.info(f"Found {len(image_files_in_zip)} image files in zip")
                slideshow.total_images = len(image_files_in_zip)
                db.session.commit()
                
                if len(image_files_in_zip) == 0:
                    raise ValueError("No image files found in zip archive")
                
                # Extract only image files
                for file_info in zip_ref.infolist():
                    if file_info.filename in image_files_in_zip:
                        # Sanitize filename to avoid directory traversal
                        safe_filename = os.path.basename(file_info.filename)
                        if safe_filename:  # Skip if empty after basename
                            zip_ref.extract(file_info, temp_extract_path)
                            # Rename to safe filename if needed
                            extracted_path = os.path.join(temp_extract_path, file_info.filename)
                            safe_path = os.path.join(temp_extract_path, safe_filename)
                            if extracted_path != safe_path:
                                os.rename(extracted_path, safe_path)
                            image_files.append(safe_path)
            
            emit_progress_update(slideshow_id, 'processing', 15, f'Extracted {len(image_files)} images')

            # Convert HEIC files to JPEG
            converted_image_files = []
            for image_file in image_files:
                if image_file.lower().endswith(('.heic', '.heif')):
                    new_path = convert_heic_to_jpeg(image_file)
                    if new_path:
                        converted_image_files.append(new_path)
                else:
                    converted_image_files.append(image_file)
            image_files = converted_image_files
            
            # Step 2: Process images with face-aware cropping
            logger.info(f"Processing {len(image_files)} images with face detection")
            emit_progress_update(slideshow_id, 'processing', 20, 'Starting image processing...')
            
            cropper = FaceAwareCropper()
            processed_images = []
            
            for i, image_file in enumerate(image_files):
                try:
                    # Calculate progress
                    progress = 20 + int((i / len(image_files)) * 60)  # 20% to 80%
                    
                    # Generate output filename
                    input_filename = os.path.basename(image_file)
                    name, ext = os.path.splitext(input_filename)
                    output_filename = f"processed_{i+1:04d}_{name}{ext}"
                    output_file_path = os.path.join(output_path, output_filename)
                    
                    # Emit progress update for current image being processed
                    emit_progress_update(
                        slideshow_id, 'processing', progress,
                        f'Processing image {i+1}/{len(image_files)}: {input_filename}',
                        total_images=len(image_files),
                        processed_images=slideshow.processed_images,
                        current_image=input_filename
                    )
                    
                    # Process single image with face detection
                    success = cropper.crop_image(
                        image_file,
                        output_file_path,
                        slideshow.screen_width,
                        slideshow.screen_height,
                        verbose=False
                    )
                    
                    if success:
                        processed_images.append(output_filename)
                        slideshow.processed_images += 1
                        logger.info(f"Successfully processed image {i+1}/{len(image_files)}: {input_filename}")
                    else:
                        logger.warning(f"Failed to process image: {input_filename}")
                    
                    # Update progress
                    slideshow.progress = progress
                    db.session.commit()
                    
                    # Emit completion update for this image
                    status_msg = "✓ Processed" if success else "⚠ Skipped"
                    emit_progress_update(
                        slideshow_id, 'processing', progress,
                        f'{status_msg} {i+1}/{len(image_files)}: {input_filename}',
                        total_images=len(image_files),
                        processed_images=slideshow.processed_images,
                        current_image=input_filename
                    )
                    
                except Exception as e:
                    logger.error(f"Error processing image {image_file}: {str(e)}")
                    # Emit error update for this image
                    emit_progress_update(
                        slideshow_id, 'processing', progress,
                        f'❌ Error processing {input_filename}: {str(e)[:50]}...',
                        total_images=len(image_files),
                        processed_images=slideshow.processed_images,
                        current_image=input_filename
                    )
                    continue
            
            if len(processed_images) == 0:
                raise ValueError("No images were successfully processed")
            
            logger.info(f"Successfully processed {len(processed_images)} images")
            emit_progress_update(slideshow_id, 'processing', 80, f'Processed {len(processed_images)} images')
            
            # Step 3: Generate slideshow HTML
            logger.info("Generating slideshow HTML")
            emit_progress_update(slideshow_id, 'processing', 85, 'Generating slideshow...')
            
            html_path = generate_slideshow_html(
                processed_images=processed_images,
                output_dir=output_path,
                zip_code=slideshow.weather_zip or "10001",
                api_key=slideshow.weather_api_key or "YOUR_API_KEY_HERE",
                screen_width=slideshow.screen_width,
                screen_height=slideshow.screen_height
            )
            
            logger.info(f"Generated slideshow HTML: {html_path}")
            emit_progress_update(slideshow_id, 'processing', 90, 'Slideshow generated')
            
            # Step 4: Clean up temporary files and uploaded zip
            logger.info("Cleaning up temporary files")
            emit_progress_update(slideshow_id, 'processing', 95, 'Cleaning up...')
            
            # Remove temporary extraction directory
            if os.path.exists(temp_extract_path):
                shutil.rmtree(temp_extract_path)
            
            # Remove uploaded zip file to save space
            if os.path.exists(upload_path):
                os.remove(upload_path)
            
            # Step 5: Mark as completed
            slideshow.status = 'completed'
            slideshow.progress = 100
            slideshow.completed_at = datetime.now(timezone.utc)
            db.session.commit()
            
            emit_progress_update(slideshow_id, 'completed', 100, 'Processing completed successfully!')
            
            logger.info(f"Slideshow processing completed for slideshow_id: {slideshow_id}")
            
            return {
                'slideshow_id': slideshow_id,
                'status': 'completed',
                'processed_images': len(processed_images),
                'html_path': html_path
            }
            
        except Exception as e:
            logger.error(f"Error processing slideshow {slideshow_id}: {str(e)}")
            
            # Update slideshow status to error
            slideshow.status = 'error'
            slideshow.error_message = str(e)
            db.session.commit()
            
            # Clean up on error
            try:
                if os.path.exists(temp_extract_path):
                    shutil.rmtree(temp_extract_path)
                # Keep upload file on error for potential retry
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup: {str(cleanup_error)}")
            
            emit_progress_update(slideshow_id, 'error', slideshow.progress, f'Error: {str(e)}')
            
            return {'error': str(e), 'slideshow_id': slideshow_id}


from flask_socketio import SocketIO

# Create a new SocketIO instance for Celery worker
socketio_celery = SocketIO(message_queue=Config.CELERY_BROKER_URL)

def emit_progress_update(slideshow_id, status, progress, message, **kwargs):
    """
    Emit progress update via WebSocket if available.
    """
    try:
        update_data = {
            'slideshow_id': slideshow_id,
            'status': status,
            'progress': progress,
            'message': message
        }
        # Add any additional data
        update_data.update(kwargs)
        
        socketio_celery.emit('progress_update', update_data, room=f'slideshow_{slideshow_id}')
        logger.info(f"Emitted progress update for slideshow {slideshow_id}: {progress}% - {message}")
    except Exception as e:
        logger.error(f"Could not emit progress update: {e}")


@celery.task(bind=True, base=DatabaseTask)
def delete_slideshow_files(self, slideshow_id):
    """
    Background task to delete slideshow files from disk.
    Called when admin deletes a slideshow.
    """
    logger.info(f"Deleting files for slideshow_id: {slideshow_id}")
    
    with self.db_session.app_context():
        slideshow = Slideshow.query.get(slideshow_id)
        if not slideshow:
            logger.error(f"Slideshow {slideshow_id} not found")
            return {'error': 'Slideshow not found'}
        
        try:
            # Delete slideshow folder
            output_path = os.path.join(Config.SLIDESHOW_FOLDER, slideshow.folder_name)
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
                logger.info(f"Deleted slideshow folder: {output_path}")
            
            # Delete any remaining upload files
            if slideshow.zip_filename:
                upload_path = os.path.join(Config.UPLOAD_FOLDER, slideshow.zip_filename)
                if os.path.exists(upload_path):
                    os.remove(upload_path)
                    logger.info(f"Deleted upload file: {upload_path}")
            
            return {'status': 'deleted', 'slideshow_id': slideshow_id}
            
        except Exception as e:
            logger.error(f"Error deleting slideshow files {slideshow_id}: {str(e)}")
            return {'error': str(e), 'slideshow_id': slideshow_id}


@celery.task(bind=True, base=DatabaseTask)
def process_folder_slideshow(self, slideshow_id):
    """
    Background task to process a slideshow from folder uploads.
    
    Steps:
    1. Scan folder for image files
    2. Process images with face-aware cropping
    3. Generate slideshow HTML
    4. Update database with results
    """
    
    logger.info(f"Starting folder slideshow processing for slideshow_id: {slideshow_id}")
    
    with self.db_session.app_context():
        # Get slideshow from database
        slideshow = Slideshow.query.get(slideshow_id)
        if not slideshow:
            logger.error(f"Slideshow {slideshow_id} not found")
            return {'error': 'Slideshow not found'}
        
        try:
            # Update status to processing
            slideshow.status = 'processing'
            slideshow.progress = 0
            db.session.commit()
            
            # Emit progress update via WebSocket (if available)
            emit_progress_update(slideshow_id, 'processing', 0, 'Starting processing...')
            
            # Define paths
            input_path = os.path.join(Config.SLIDESHOW_FOLDER, slideshow.folder_name)
            
            if not os.path.exists(input_path):
                raise ValueError(f"Slideshow folder not found: {input_path}")
            
            # Step 1: Scan folder for image files
            logger.info(f"Scanning folder for images: {input_path}")
            emit_progress_update(slideshow_id, 'processing', 5, 'Scanning for images...')
            
            supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.heic', '.heif'}
            image_files = []
            
            # Recursively find all image files
            for root, dirs, files in os.walk(input_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in supported_extensions):
                        full_path = os.path.join(root, file)
                        image_files.append(full_path)
            
            logger.info(f"Found {len(image_files)} image files in folder")
            
            # Convert HEIC files to JPEG
            converted_image_files = []
            for image_file in image_files:
                if image_file.lower().endswith(('.heic', '.heif')):
                    new_path = convert_heic_to_jpeg(image_file)
                    if new_path:
                        converted_image_files.append(new_path)
                else:
                    converted_image_files.append(image_file)
            image_files = converted_image_files
            
            slideshow.total_images = len(image_files)
            db.session.commit()
            
            if len(image_files) == 0:
                raise ValueError("No image files found in uploaded folder")
            
            emit_progress_update(slideshow_id, 'processing', 15, f'Found {len(image_files)} images')
            
            # Step 2: Process images with face-aware cropping
            logger.info(f"Processing {len(image_files)} images with face detection")
            emit_progress_update(slideshow_id, 'processing', 20, 'Starting image processing...')
            
            cropper = FaceAwareCropper()
            processed_images = []
            
            for i, image_file in enumerate(image_files):
                try:
                    # Calculate progress
                    progress = 20 + int((i / len(image_files)) * 60)  # 20% to 80%
                    
                    # Generate output filename
                    input_filename = os.path.basename(image_file)
                    name, ext = os.path.splitext(input_filename)
                    output_filename = f"processed_{i+1:04d}_{name}{ext}"
                    output_file_path = os.path.join(input_path, output_filename)
                    
                    # Emit progress update for current image being processed
                    emit_progress_update(
                        slideshow_id, 'processing', progress,
                        f'Processing image {i+1}/{len(image_files)}: {input_filename}',
                        total_images=len(image_files),
                        processed_images=slideshow.processed_images,
                        current_image=input_filename
                    )
                    
                    # Process single image with face detection
                    success = cropper.crop_image(
                        image_file,
                        output_file_path,
                        slideshow.screen_width,
                        slideshow.screen_height,
                        verbose=False
                    )
                    
                    if success:
                        processed_images.append(output_filename)
                        slideshow.processed_images += 1
                        logger.info(f"Successfully processed image {i+1}/{len(image_files)}: {input_filename}")
                        
                        # Remove original file to save space (if it's different from output)
                        if image_file != output_file_path and os.path.exists(image_file):
                            os.remove(image_file)
                    else:
                        logger.warning(f"Failed to process image: {input_filename}")
                    
                    # Update progress
                    slideshow.progress = progress
                    db.session.commit()
                    
                    # Emit completion update for this image
                    status_msg = "✓ Processed" if success else "⚠ Skipped"
                    emit_progress_update(
                        slideshow_id, 'processing', progress,
                        f'{status_msg} {i+1}/{len(image_files)}: {input_filename}',
                        total_images=len(image_files),
                        processed_images=slideshow.processed_images,
                        current_image=input_filename
                    )
                    
                except Exception as e:
                    logger.error(f"Error processing image {image_file}: {str(e)}")
                    # Emit error update for this image
                    emit_progress_update(
                        slideshow_id, 'processing', progress,
                        f'❌ Error processing {input_filename}: {str(e)[:50]}...',
                        total_images=len(image_files),
                        processed_images=slideshow.processed_images,
                        current_image=input_filename
                    )
                    continue
            
            if len(processed_images) == 0:
                raise ValueError("No images were successfully processed")
            
            logger.info(f"Successfully processed {len(processed_images)} images")
            emit_progress_update(slideshow_id, 'processing', 80, f'Processed {len(processed_images)} images')
            
            # Step 3: Generate slideshow HTML
            logger.info("Generating slideshow HTML")
            emit_progress_update(slideshow_id, 'processing', 85, 'Generating slideshow...')
            
            html_path = generate_slideshow_html(
                processed_images=processed_images,
                output_dir=input_path,
                zip_code=slideshow.weather_zip or "10001",
                api_key=slideshow.weather_api_key or "YOUR_API_KEY_HERE",
                screen_width=slideshow.screen_width,
                screen_height=slideshow.screen_height
            )
            
            logger.info(f"Generated slideshow HTML: {html_path}")
            emit_progress_update(slideshow_id, 'processing', 90, 'Slideshow generated')
            
            # Step 4: Mark as completed
            slideshow.status = 'completed'
            slideshow.progress = 100
            slideshow.completed_at = datetime.now(timezone.utc)
            db.session.commit()
            
            emit_progress_update(slideshow_id, 'completed', 100, 'Processing completed successfully!')
            
            logger.info(f"Folder slideshow processing completed for slideshow_id: {slideshow_id}")
            
            return {
                'slideshow_id': slideshow_id,
                'status': 'completed',
                'processed_images': len(processed_images),
                'html_path': html_path
            }
            
        except Exception as e:
            logger.error(f"Error processing folder slideshow {slideshow_id}: {str(e)}")

            # Update slideshow status to error
            slideshow.status = 'error'
            slideshow.error_message = str(e)
            db.session.commit()

            emit_progress_update(slideshow_id, 'error', slideshow.progress, f'Error: {str(e)}')

            return {'error': str(e), 'slideshow_id': slideshow_id}


@celery.task(bind=True, base=DatabaseTask)
def add_photos_to_slideshow(self, slideshow_id):
    """
    Background task to add new photos to an existing slideshow.

    Steps:
    1. Scan existing slideshow folder for processed images
    2. Process new images with face-aware cropping (numbered sequentially after existing)
    3. Regenerate slideshow HTML with all images
    4. Update database with new totals
    """

    logger.info(f"Starting to add photos to slideshow_id: {slideshow_id}")

    with self.db_session.app_context():
        # Get slideshow from database
        slideshow = Slideshow.query.get(slideshow_id)
        if not slideshow:
            logger.error(f"Slideshow {slideshow_id} not found")
            return {'error': 'Slideshow not found'}

        try:
            # Update status to processing
            slideshow.status = 'processing'
            slideshow.progress = 0
            db.session.commit()

            # Emit progress update via WebSocket (if available)
            emit_progress_update(slideshow_id, 'processing', 0, 'Starting to add photos...')

            # Define paths
            slideshow_path = os.path.join(Config.SLIDESHOW_FOLDER, slideshow.folder_name)

            if not os.path.exists(slideshow_path):
                raise ValueError(f"Slideshow folder not found: {slideshow_path}")

            # Step 1: Find existing processed images to determine starting index
            logger.info(f"Scanning for existing processed images: {slideshow_path}")
            emit_progress_update(slideshow_id, 'processing', 5, 'Scanning existing images...')

            existing_processed = []
            new_unprocessed = []

            for file in os.listdir(slideshow_path):
                if file.startswith('processed_') and not file.endswith('.html'):
                    existing_processed.append(file)
                elif file != 'slideshow.html':
                    # Check if it's an image file that needs processing
                    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.heic', '.heif'}
                    if any(file.lower().endswith(ext) for ext in supported_extensions):
                        new_unprocessed.append(os.path.join(slideshow_path, file))

            # Convert HEIC files to JPEG
            converted_unprocessed = []
            for image_file in new_unprocessed:
                if image_file.lower().endswith(('.heic', '.heif')):
                    new_path = convert_heic_to_jpeg(image_file)
                    if new_path:
                        converted_unprocessed.append(new_path)
                else:
                    converted_unprocessed.append(image_file)
            new_unprocessed = converted_unprocessed

            # Determine starting index
            start_index = len(existing_processed) + 1
            logger.info(f"Found {len(existing_processed)} existing processed images, starting at index {start_index}")
            logger.info(f"Found {len(new_unprocessed)} new images to process")

            if len(new_unprocessed) == 0:
                raise ValueError("No new images found to add")

            # Update total image count
            slideshow.total_images = len(existing_processed) + len(new_unprocessed)
            db.session.commit()

            emit_progress_update(slideshow_id, 'processing', 15, f'Found {len(new_unprocessed)} new images to add')

            # Step 2: Process new images with face-aware cropping
            logger.info(f"Processing {len(new_unprocessed)} new images with face detection")
            emit_progress_update(slideshow_id, 'processing', 20, 'Starting image processing...')

            cropper = FaceAwareCropper()
            newly_processed = []

            for i, image_file in enumerate(new_unprocessed):
                try:
                    # Calculate progress
                    progress = 20 + int((i / len(new_unprocessed)) * 60)  # 20% to 80%

                    # Generate output filename with sequential numbering
                    input_filename = os.path.basename(image_file)
                    name, ext = os.path.splitext(input_filename)
                    output_filename = f"processed_{start_index + i:04d}_{name}{ext}"
                    output_file_path = os.path.join(slideshow_path, output_filename)

                    # Emit progress update for current image being processed
                    emit_progress_update(
                        slideshow_id, 'processing', progress,
                        f'Processing image {i+1}/{len(new_unprocessed)}: {input_filename}',
                        total_images=slideshow.total_images,
                        processed_images=len(existing_processed) + len(newly_processed),
                        current_image=input_filename
                    )

                    # Process single image with face detection
                    success = cropper.crop_image(
                        image_file,
                        output_file_path,
                        slideshow.screen_width,
                        slideshow.screen_height,
                        verbose=False
                    )

                    if success:
                        newly_processed.append(output_filename)
                        logger.info(f"Successfully processed image {i+1}/{len(new_unprocessed)}: {input_filename}")

                        # Remove original file to save space
                        if os.path.exists(image_file):
                            os.remove(image_file)
                    else:
                        logger.warning(f"Failed to process image: {input_filename}")

                    # Update progress
                    slideshow.progress = progress
                    slideshow.processed_images = len(existing_processed) + len(newly_processed)
                    db.session.commit()

                    # Emit completion update for this image
                    status_msg = "✓ Processed" if success else "⚠ Skipped"
                    emit_progress_update(
                        slideshow_id, 'processing', progress,
                        f'{status_msg} {i+1}/{len(new_unprocessed)}: {input_filename}',
                        total_images=slideshow.total_images,
                        processed_images=slideshow.processed_images,
                        current_image=input_filename
                    )

                except Exception as e:
                    logger.error(f"Error processing image {image_file}: {str(e)}")
                    # Emit error update for this image
                    emit_progress_update(
                        slideshow_id, 'processing', progress,
                        f'❌ Error processing {input_filename}: {str(e)[:50]}...',
                        total_images=slideshow.total_images,
                        processed_images=slideshow.processed_images,
                        current_image=input_filename
                    )
                    continue

            if len(newly_processed) == 0:
                raise ValueError("No new images were successfully processed")

            logger.info(f"Successfully processed {len(newly_processed)} new images")
            emit_progress_update(slideshow_id, 'processing', 80, f'Processed {len(newly_processed)} new images')

            # Step 3: Regenerate slideshow HTML with all images
            logger.info("Regenerating slideshow HTML with all images")
            emit_progress_update(slideshow_id, 'processing', 85, 'Regenerating slideshow...')

            # Get all processed images (existing + new)
            all_processed = sorted(existing_processed + newly_processed)

            html_path = generate_slideshow_html(
                processed_images=all_processed,
                output_dir=slideshow_path,
                zip_code=slideshow.weather_zip or "10001",
                api_key=slideshow.weather_api_key or "YOUR_API_KEY_HERE",
                screen_width=slideshow.screen_width,
                screen_height=slideshow.screen_height
            )

            logger.info(f"Regenerated slideshow HTML: {html_path}")
            emit_progress_update(slideshow_id, 'processing', 90, 'Slideshow regenerated')

            # Step 4: Update database with final counts
            slideshow.status = 'completed'
            slideshow.progress = 100
            slideshow.total_images = len(all_processed)
            slideshow.processed_images = len(all_processed)
            db.session.commit()

            emit_progress_update(slideshow_id, 'completed', 100, f'Successfully added {len(newly_processed)} photos!')

            logger.info(f"Successfully added {len(newly_processed)} photos to slideshow_id: {slideshow_id}")

            return {
                'slideshow_id': slideshow_id,
                'status': 'completed',
                'newly_processed_images': len(newly_processed),
                'total_images': len(all_processed),
                'html_path': html_path
            }

        except Exception as e:
            logger.error(f"Error adding photos to slideshow {slideshow_id}: {str(e)}")

            # Update slideshow status to error
            slideshow.status = 'error'
            slideshow.error_message = str(e)
            db.session.commit()

            emit_progress_update(slideshow_id, 'error', slideshow.progress, f'Error: {str(e)}')

            return {'error': str(e), 'slideshow_id': slideshow_id}