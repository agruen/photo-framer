"""WSGI entry point for Gunicorn"""
from app.app import create_app, socketio

# Create the Flask application
app = create_app()

if __name__ == "__main__":
    # This is used when running with Gunicorn
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)