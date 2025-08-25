# Gunicorn configuration for handling large file uploads
import multiprocessing

# Server socket
bind = "0.0.0.0:5000"
backlog = 2048

# Worker processes
workers = 1  # Use 1 worker to avoid memory issues on Pi
worker_class = "eventlet"  # Use eventlet for WebSocket support
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
preload_app = True

# Timeouts - Extended for large file uploads
timeout = 1800  # 30 minutes for large uploads
keepalive = 2
graceful_timeout = 30

# Memory and file limits
worker_tmp_dir = "/dev/shm"  # Use RAM for temporary files
max_requests = 0  # Disable worker recycling to avoid interrupting uploads

# Logging
loglevel = "info"
accesslog = "-"
errorlog = "-"

# Process naming
proc_name = "photo-framer"

# Preload modules
def when_ready(server):
    server.log.info("Photo Framer server ready. Listening on %s", server.address)