"""
Setup utility for first-time configuration
"""
import os
import json
from pathlib import Path

CONFIG_DIR = '/app/db'
CONFIG_FILE = os.path.join(CONFIG_DIR, 'config.json')
SETUP_COMPLETE_FILE = os.path.join(CONFIG_DIR, '.setup_complete')


def is_setup_complete():
    """Check if initial setup has been completed"""
    return os.path.exists(SETUP_COMPLETE_FILE)


def get_config():
    """Load configuration from file"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_config(config_data):
    """Save configuration to file"""
    # Ensure config directory exists
    os.makedirs(CONFIG_DIR, exist_ok=True)

    # Load existing config and update
    existing_config = get_config()
    existing_config.update(config_data)

    # Write config file
    with open(CONFIG_FILE, 'w') as f:
        json.dump(existing_config, f, indent=2)

    return existing_config


def mark_setup_complete():
    """Mark setup as complete"""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    Path(SETUP_COMPLETE_FILE).touch()


def get_admin_password():
    """Get admin password from config or environment"""
    # Check environment variable first
    env_password = os.environ.get('ADMIN_PASSWORD')
    if env_password:
        return env_password

    # Then check config file
    config = get_config()
    return config.get('admin_password', 'admin123')


def get_secret_key():
    """Get secret key from config or environment"""
    # Check environment variable first
    env_key = os.environ.get('SECRET_KEY')
    if env_key:
        return env_key

    # Then check config file
    config = get_config()
    if 'secret_key' in config:
        return config['secret_key']

    # Generate a new one if not found
    import secrets
    new_key = secrets.token_hex(32)
    save_config({'secret_key': new_key})
    return new_key


def get_default_settings():
    """Get default settings from config"""
    config = get_config()
    return {
        'screen_width': config.get('default_screen_width', 1280),
        'screen_height': config.get('default_screen_height', 800),
        'rotation_interval': config.get('default_rotation_interval', 60)
    }
