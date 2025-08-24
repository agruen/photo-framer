#!/usr/bin/env python3
"""
Test script to validate Docker server setup without running the full stack.
Tests configuration files, imports, and basic functionality.
"""

import os
import sys
import json
import tempfile
import importlib.util
from pathlib import Path

def test_environment_files():
    """Test that configuration files are properly set up."""
    print("🔧 Testing configuration files...")
    
    # Check for required files
    required_files = [
        'docker-compose.yml',
        'Dockerfile', 
        'requirements-server.txt',
        '.env.example',
        'docker-compose.override.yml.example'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All configuration files present")
    return True

def test_app_structure():
    """Test that the Flask app structure is correct."""
    print("🏗️  Testing application structure...")
    
    # Check app directory structure
    app_files = [
        'app/__init__.py',
        'app/app.py',
        'app/models.py', 
        'app/tasks.py',
        'app/config.py',
        'app/celery_app.py'
    ]
    
    template_files = [
        'app/templates/base.html',
        'app/templates/login.html',
        'app/templates/dashboard.html',
        'app/templates/create.html',
        'app/templates/slideshow_detail.html'
    ]
    
    all_files = app_files + template_files
    missing_files = []
    
    for file in all_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing app files: {missing_files}")
        return False
    
    print("✅ Application structure is complete")
    return True

def test_python_imports():
    """Test that Python modules can be imported without missing dependencies."""
    print("🐍 Testing Python module imports...")
    
    # Test if we can import the slideshow generator
    try:
        import slideshow_generator
        print("✅ slideshow_generator imports successfully")
    except ImportError as e:
        print(f"❌ slideshow_generator import failed: {e}")
        return False
    
    # Test slideshow HTML generation
    try:
        html_path = slideshow_generator.generate_slideshow_html(
            processed_images=['test1.jpg', 'test2.png'],
            output_dir=tempfile.gettempdir(),
            zip_code="10001", 
            api_key="test_key",
            screen_width=1280,
            screen_height=800
        )
        
        # Check that HTML was generated
        if os.path.exists(html_path):
            with open(html_path, 'r') as f:
                html_content = f.read()
            
            # Basic validation
            if 'test1.jpg' in html_content and 'test2.png' in html_content:
                print("✅ Slideshow HTML generation works")
                os.remove(html_path)  # Clean up
            else:
                print("❌ Generated HTML doesn't contain expected images")
                return False
        else:
            print("❌ HTML file was not generated")
            return False
            
    except Exception as e:
        print(f"❌ Slideshow generation failed: {e}")
        return False
    
    return True

def test_docker_compose_syntax():
    """Test Docker Compose file syntax."""
    print("🐳 Testing Docker Compose configuration...")
    
    try:
        import yaml
        
        with open('docker-compose.yml', 'r') as f:
            compose_config = yaml.safe_load(f)
        
        # Check required services
        required_services = ['web', 'worker', 'redis']
        if 'services' not in compose_config:
            print("❌ No services defined in docker-compose.yml")
            return False
        
        missing_services = []
        for service in required_services:
            if service not in compose_config['services']:
                missing_services.append(service)
        
        if missing_services:
            print(f"❌ Missing Docker services: {missing_services}")
            return False
        
        print("✅ Docker Compose configuration is valid")
        return True
        
    except ImportError:
        print("⚠️  PyYAML not available, skipping Docker Compose syntax check")
        return True
    except Exception as e:
        print(f"❌ Docker Compose configuration error: {e}")
        return False

def test_volume_directories():
    """Test that volume directories can be created."""
    print("📁 Testing volume directory structure...")
    
    volume_dirs = [
        'volumes/uploads',
        'volumes/slideshows', 
        'volumes/db',
        'volumes/redis',
        'volumes/temp'
    ]
    
    try:
        for vol_dir in volume_dirs:
            os.makedirs(vol_dir, exist_ok=True)
        
        print("✅ Volume directories created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create volume directories: {e}")
        return False

def test_requirements():
    """Check that requirements files are readable."""
    print("📦 Testing requirements files...")
    
    req_files = ['requirements.txt', 'requirements-server.txt']
    
    for req_file in req_files:
        if not os.path.exists(req_file):
            print(f"❌ Missing requirements file: {req_file}")
            return False
        
        try:
            with open(req_file, 'r') as f:
                requirements = f.read()
            
            # Check for key dependencies
            if req_file == 'requirements-server.txt':
                required_packages = ['Flask', 'celery', 'redis', 'opencv-python']
                for package in required_packages:
                    if package.lower() not in requirements.lower():
                        print(f"❌ Missing required package '{package}' in {req_file}")
                        return False
        
        except Exception as e:
            print(f"❌ Error reading {req_file}: {e}")
            return False
    
    print("✅ Requirements files are valid")
    return True

def main():
    """Run all validation tests."""
    print("🧪 Photo Frame Server Setup Validation")
    print("=" * 40)
    
    tests = [
        test_environment_files,
        test_app_structure, 
        test_python_imports,
        test_docker_compose_syntax,
        test_volume_directories,
        test_requirements
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()  # Add spacing between tests
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            print()
    
    print("=" * 40)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Server setup is ready.")
        print()
        print("🚀 Next steps:")
        print("1. Run: ./start-server.sh")
        print("2. Open: http://localhost:5000")
        print("3. Login with default password: changeme123")
        return True
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)