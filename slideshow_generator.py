#!/usr/bin/env python3
"""
Slideshow Generator for Photo Frame

This module generates HTML slideshows with weather and time display,
designed to work with the existing face_crop_tool.py cropping functionality.
"""

import json
import os


def generate_slideshow_html(processed_images, output_dir, zip_code, api_key, screen_width=1280, screen_height=800):
    """
    Generate a full-screen HTML slideshow with weather and time display.
    
    Creates a responsive HTML file that displays processed images in a slideshow format
    with integrated weather information and digital clock. The slideshow is optimized
    for the configured screen resolution and includes the following features:
    
    - Random image rotation every 60 seconds
    - Real-time clock display (12-hour format)
    - Weather information with icons and temperature
    - Full-screen display optimized for digital photo frames
    - Automatic weather data refresh every 4 minutes
    
    Args:
        processed_images (list): List of processed image filenames to include in slideshow
        output_dir (str): Directory where the HTML file will be saved
        zip_code (str): US ZIP code for weather data retrieval
        api_key (str): OpenWeatherMap API key for weather data access
        screen_width (int): Target screen width in pixels
        screen_height (int): Target screen height in pixels
    
    Returns:
        str: Absolute path to the generated HTML slideshow file
        
    Note:
        The weather functionality requires an internet connection and uses the
        OpenWeatherMap API. The API key is embedded in the HTML for demonstration
        purposes - in production, consider using environment variables or a backend service.
    """
    
    # Convert image filenames to relative paths for HTML slideshow
    # Filter out any None values from failed image processing
    image_list = [f"./{img}" for img in processed_images if img is not None]
    # Convert to JSON format for embedding in JavaScript
    image_list_json = json.dumps(image_list)
    
    html_template = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Photo Slideshow - {screen_width}x{screen_height}</title>
  <style>
    body {{
        margin: 0;
        padding: 0;
        overflow: hidden;
        width: {screen_width}px;
        height: {screen_height}px;
    }}

    #slideshow {{
        width: {screen_width}px;
        height: {screen_height}px;
        position: fixed;
        top: 0;
        left: 0;
        background-color: lightgray;
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        z-index: -1;
    }}

    #clock {{
        position: fixed;
        bottom: 30px;
        left: 30px;
        font-family: Arial, sans-serif;
        font-size: 64px;
        font-weight: bold;
        color: white;
        z-index: 100;
        text-shadow: 2px 0 0 #000, 0 -2px 0 #000, 0 2px 0 #000, -2px 0 0 #000;
    }}

    #weather {{
        position: fixed;
        bottom: 30px;
        right: 30px;
        font-family: Arial, sans-serif;
        font-size: 64px;
        font-weight: bold;
        color: white;
        z-index: 100;
        text-shadow: 2px 0 0 #000, 0 -2px 0 #000, 0 2px 0 #000, -2px 0 0 #000;
    }}

    #weather img {{
        vertical-align: middle;
        width: 80px;
        height: 80px;
    }}
  </style>
</head>
<body>
  <div id="slideshow"></div>
  <div id="clock"></div>
  <div id="weather"></div>

  <script>
    // Array of processed image paths for slideshow rotation
    var images = {image_list_json};
    // Track when weather was last updated (rate limiting)
    var lastWeatherUpdate = 0;

    // Select a random image from the available collection
    function getRandomImage(images) {{
      var index = Math.floor(Math.random() * images.length);
      return images[index];
    }}

    // Update the background image with a random selection
    function updateSlideshow(images) {{
      var slideshow = document.getElementById("slideshow");
      slideshow.style.backgroundImage = "url('" + getRandomImage(images) + "')";
    }}

    // Update the digital clock display with current time
    function updateClock() {{
      var clock = document.getElementById("clock");
      var currentDate = new Date();
      var hours = currentDate.getHours();
      var minutes = currentDate.getMinutes();
      var period = hours >= 12 ? "PM" : "AM";

      // Convert from 24-hour to 12-hour format
      hours = hours % 12;
      hours = hours ? hours : 12;  // Handle midnight (0 hours = 12 AM)

      // Add leading zero to minutes for consistent formatting
      minutes = minutes < 10 ? "0" + minutes : minutes;

      // Display formatted time
      clock.innerHTML = hours + ":" + minutes + " " + period;
    }}

    // Fetch and display current weather information
    function updateWeather(zipCode) {{
      var weatherDiv = document.getElementById("weather");
      // OpenWeatherMap API configuration
      var apiKey = '{api_key}';
      
      // Check if API key is configured
      if (apiKey === 'YOUR_API_KEY_HERE' || !apiKey || apiKey.length < 10) {{
        weatherDiv.innerHTML = "Configure API key";
        console.error('Weather API key not configured. Edit WEATHER_API_KEY in the script.');
        return;
      }}
      
      var url = `https://api.openweathermap.org/data/2.5/weather?zip=${{zipCode}}&units=imperial&appid=${{apiKey}}`;
      console.log('Weather API URL (without key):', url.replace(apiKey, '[HIDDEN]'));

      var currentTime = new Date().getTime();
      // Only update weather every 5 minutes (300000ms) to avoid rate limiting
      if (currentTime - lastWeatherUpdate >= 300000) {{
        console.log('Fetching weather data...');
        weatherDiv.innerHTML = "Loading weather...";
        
        fetch(url)
          .then(response => {{
            console.log('Weather API response status:', response.status);
            if (!response.ok) {{
              throw new Error(`HTTP ${{response.status}}: ${{response.statusText}}`);
            }}
            return response.json();
          }})
          .then(data => {{
            console.log('Weather data received:', data);
            // Check if response has expected structure
            if (!data.main || !data.weather || !data.weather[0]) {{
              throw new Error('Invalid weather data structure');
            }}
            // Extract temperature and weather icon from API response
            var temperature = Math.round(data.main.temp);
            var icon = data.weather[0].icon;
            var iconUrl = `https://openweathermap.org/img/wn/${{icon}}@2x.png`;
            // Display weather icon and temperature
            weatherDiv.innerHTML = `<img src="${{iconUrl}}" alt="weather icon"> ${{temperature}}°F`;
            lastWeatherUpdate = currentTime;
            console.log('Weather updated successfully');
          }})
          .catch(error => {{
            // Handle API errors gracefully with detailed error info
            console.error('Weather API error:', error);
            var errorMsg = "Weather unavailable";
            if (error.message.includes('401')) {{
              errorMsg = "Invalid API key";
            }} else if (error.message.includes('404')) {{
              errorMsg = "Invalid ZIP code";
            }} else if (error.message.includes('429')) {{
              errorMsg = "Rate limit exceeded";
            }}
            weatherDiv.innerHTML = errorMsg;
          }});
      }} else {{
        console.log('Weather update skipped (too soon)');
      }}
    }}

    // Initialize and start the slideshow with all interactive elements
    function startSlideshow(zipCode) {{
      // Only start image rotation if we have images to display
      if (images.length > 0) {{
        // Change slideshow image every 60 seconds (60000ms)
        setInterval(function() {{
          updateSlideshow(images);
        }}, 60000);
      }}
      // Update clock every second for real-time display
      setInterval(updateClock, 1000);
      // Check for weather updates every 5 minutes (300000ms)
      setInterval(function() {{
        updateWeather(zipCode);
      }}, 300000);
      
      console.log('Slideshow initialized with ZIP code:', zipCode);
      
      // Initialize display elements immediately
      updateWeather(zipCode);   // Get initial weather data
      updateClock();           // Show current time
      updateSlideshow(images); // Display first image
    }}

    startSlideshow('{zip_code}');
  </script>
</body>
</html>"""

    # Write the HTML file to the output directory
    html_path = os.path.join(output_dir, 'slideshow.html')
    with open(html_path, 'w') as file:
        file.write(html_template)
    
    return html_path


def create_slideshow_from_folder(output_folder, zip_code, api_key, screen_width=1280, screen_height=800):
    """
    Create a slideshow from all images in the specified output folder.
    
    Args:
        output_folder (str): Path to folder containing processed images
        zip_code (str): ZIP code for weather data
        api_key (str): OpenWeatherMap API key
        screen_width (int): Target screen width
        screen_height (int): Target screen height
    
    Returns:
        str: Path to the generated HTML slideshow file
    """
    
    if not os.path.exists(output_folder):
        print(f"Error: Output folder '{output_folder}' does not exist")
        return None
    
    # Find all image files in the output folder
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for filename in os.listdir(output_folder):
        if any(filename.lower().endswith(ext) for ext in supported_extensions):
            image_files.append(filename)
    
    if not image_files:
        print(f"No image files found in '{output_folder}'")
        return None
    
    print(f"Found {len(image_files)} images for slideshow")
    
    # Generate the slideshow HTML
    html_path = generate_slideshow_html(
        image_files, output_folder, zip_code, api_key, screen_width, screen_height
    )
    
    return html_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate HTML slideshow from processed images')
    parser.add_argument('--folder', '-f', required=True,
                       help='Folder containing processed images')
    parser.add_argument('--zip', '-z', required=True,
                       help='ZIP code for weather data')
    parser.add_argument('--api-key', '-k', required=True,
                       help='OpenWeatherMap API key')
    parser.add_argument('--width', '-w', type=int, default=1280,
                       help='Screen width (default: 1280)')
    parser.add_argument('--height', '-ht', type=int, default=800,
                       help='Screen height (default: 800)')
    
    args = parser.parse_args()
    
    html_path = create_slideshow_from_folder(
        args.folder, args.zip, args.api_key, args.width, args.height
    )
    
    if html_path:
        print(f"Slideshow generated: {html_path}")
        print(f"Open {html_path} in a web browser to view the slideshow")