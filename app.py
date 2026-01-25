from flask import Flask, request, jsonify, send_file, Response
from PIL import Image, ImageDraw, ImageFont
import cv2
import base64
import requests
import tempfile
import os
import io
import logging
import subprocess
import shutil
import threading
import time
from typing import Optional, Dict, Any, List
from uuid import uuid4
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# Directory for storing generated videos temporarily
VIDEO_STORAGE_DIR = '/tmp/generated_videos'
os.makedirs(VIDEO_STORAGE_DIR, exist_ok=True)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # type: ignore

# Video cleanup settings
VIDEO_EXPIRY_HOURS = 1  # Videos expire after 1 hour


def cleanup_old_videos():
    """Remove videos older than VIDEO_EXPIRY_HOURS"""
    try:
        now = datetime.now()
        for filename in os.listdir(VIDEO_STORAGE_DIR):
            filepath = os.path.join(VIDEO_STORAGE_DIR, filename)
            if os.path.isfile(filepath):
                file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                if now - file_time > timedelta(hours=VIDEO_EXPIRY_HOURS):
                    os.remove(filepath)
                    logger.info(f"Cleaned up old video: {filename}")
    except Exception as e:
        logger.error(f"Cleanup error: {e}")


def start_cleanup_thread():
    """Start background thread for video cleanup"""
    def cleanup_loop():
        while True:
            cleanup_old_videos()
            time.sleep(1800)  # Run every 30 minutes
    
    thread = threading.Thread(target=cleanup_loop, daemon=True)
    thread.start()


# Start cleanup thread on app start
start_cleanup_thread()


def should_keep_frame(frame) -> bool:
    """Returns True if frame has a face >= 1/6th of image dimensions"""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        img_h, img_w = frame.shape[:2]
        for (x, y, w, h) in faces:
            if w >= (img_w / 6) or h >= (img_h / 6):
                return True
        return False
    except Exception as e:
        logger.error(f"Face detection error: {e}")
        return True


def download_image_from_url(url: str) -> Image.Image:
    """Download image from URL and return PIL Image"""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert('RGBA')


def download_image_to_file(url: str, filepath: str) -> bool:
    """Download image from URL and save to file"""
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        # Load with PIL to ensure valid image and convert to RGB
        img = Image.open(io.BytesIO(response.content))
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (0, 0, 0))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        img.save(filepath, 'JPEG', quality=95)
        return True
    except Exception as e:
        logger.error(f"Failed to download image: {e}")
        return False


def download_image_from_telegram(file_id: str, bot_token: str) -> Image.Image:
    """Download image from Telegram using file_id"""
    # Get file path
    file_info = requests.get(
        f"https://api.telegram.org/bot{bot_token}/getFile?file_id={file_id}",
        timeout=30
    ).json()
    
    if not file_info.get('ok'):
        raise Exception(f"Failed to get file info: {file_info}")
    
    file_path = file_info['result']['file_path']
    file_url = f"https://api.telegram.org/file/bot{bot_token}/{file_path}"
    
    response = requests.get(file_url, timeout=30)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert('RGBA')


def apply_logo_overlay(
    base_image: Image.Image,
    logo: Image.Image,
    position: str = 'bottom-right',
    opacity: float = 0.8,
    size_percent: float = 12
) -> Image.Image:
    """Apply logo overlay to base image"""
    
    base_width, base_height = base_image.size
    
    # Calculate logo size (percentage of image width)
    logo_width = int(base_width * (size_percent / 100))
    logo_height = int(logo.height * (logo_width / logo.width))
    
    # Resize logo
    logo = logo.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
    
    # Apply opacity to logo
    if logo.mode != 'RGBA':
        logo = logo.convert('RGBA')
    
    # Modify alpha channel for opacity
    r, g, b, a = logo.split()
    a = a.point(lambda p: int(p * opacity))
    logo = Image.merge('RGBA', (r, g, b, a))
    
    # Calculate position
    padding = int(base_width * 0.03)  # 3% padding
    
    positions = {
        'bottom-right': (base_width - logo_width - padding, base_height - logo_height - padding),
        'bottom-left': (padding, base_height - logo_height - padding),
        'top-right': (base_width - logo_width - padding, padding),
        'top-left': (padding, padding),
        'center': ((base_width - logo_width) // 2, (base_height - logo_height) // 2)
    }
    
    pos = positions.get(position, positions['bottom-right'])
    
    # Create a copy of base image and paste logo
    result = base_image.copy()
    result.paste(logo, pos, logo)
    
    return result


def get_video_dimensions(aspect_ratio: str) -> tuple:
    """Get video dimensions based on aspect ratio"""
    dimensions = {
        '9:16': (1080, 1920),   # Stories/Reels
        '16:9': (1920, 1080),   # YouTube
        '1:1': (1080, 1080),    # Square
        '4:5': (1080, 1350),    # Instagram Feed
        '4:3': (1440, 1080),    # Standard
    }
    return dimensions.get(aspect_ratio, (1080, 1920))


def create_slideshow_with_ffmpeg(
    image_paths: List[str],
    output_path: str,
    duration_per_image: float = 3.0,
    transition: str = 'fade',
    transition_duration: float = 0.5,
    fps: int = 30,
    aspect_ratio: str = '9:16'
) -> bool:
    """Create slideshow video using FFmpeg"""
    
    if len(image_paths) < 1:
        logger.error("No images provided")
        return False
    
    width, height = get_video_dimensions(aspect_ratio)
    
    try:
        if transition == 'none' or len(image_paths) == 1:
            # Simple concatenation without transitions
            return create_simple_slideshow(image_paths, output_path, duration_per_image, fps, width, height)
        elif transition == 'fade':
            return create_fade_slideshow(image_paths, output_path, duration_per_image, transition_duration, fps, width, height)
        elif transition == 'slide':
            return create_slide_slideshow(image_paths, output_path, duration_per_image, transition_duration, fps, width, height)
        elif transition == 'zoom':
            return create_zoom_slideshow(image_paths, output_path, duration_per_image, fps, width, height)
        else:
            # Default to simple
            return create_simple_slideshow(image_paths, output_path, duration_per_image, fps, width, height)
            
    except Exception as e:
        logger.error(f"FFmpeg error: {e}")
        return False


def create_simple_slideshow(
    image_paths: List[str],
    output_path: str,
    duration: float,
    fps: int,
    width: int,
    height: int
) -> bool:
    """Create slideshow without transitions"""
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create input file list
        input_file = os.path.join(temp_dir, 'input.txt')
        with open(input_file, 'w') as f:
            for path in image_paths:
                f.write(f"file '{path}'\n")
                f.write(f"duration {duration}\n")
            # Add last image again for proper ending
            f.write(f"file '{image_paths[-1]}'\n")
        
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', input_file,
            '-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,fps={fps}',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'fast',
            '-crf', '23',
            '-movflags', '+faststart',
            output_path
        ]
        
        logger.info(f"Running FFmpeg: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            return False
        
        return os.path.exists(output_path)
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def create_fade_slideshow(
    image_paths: List[str],
    output_path: str,
    duration: float,
    transition_duration: float,
    fps: int,
    width: int,
    height: int
) -> bool:
    """Create slideshow with fade transitions using xfade filter"""
    
    if len(image_paths) < 2:
        return create_simple_slideshow(image_paths, output_path, duration, fps, width, height)
    
    try:
        # Build FFmpeg command with xfade filters
        inputs = []
        for path in image_paths:
            inputs.extend(['-loop', '1', '-t', str(duration + transition_duration), '-i', path])
        
        # Build filter complex
        filter_parts = []
        
        # Scale all inputs
        for i in range(len(image_paths)):
            filter_parts.append(
                f'[{i}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,'
                f'pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps={fps}[v{i}]'
            )
        
        # Chain xfade filters
        if len(image_paths) == 2:
            offset = duration - transition_duration
            filter_parts.append(
                f'[v0][v1]xfade=transition=fade:duration={transition_duration}:offset={offset}[outv]'
            )
        else:
            # First xfade
            offset = duration - transition_duration
            filter_parts.append(
                f'[v0][v1]xfade=transition=fade:duration={transition_duration}:offset={offset}[xf0]'
            )
            
            # Middle xfades
            for i in range(2, len(image_paths) - 1):
                prev_offset = offset
                offset = prev_offset + duration - transition_duration
                filter_parts.append(
                    f'[xf{i-2}][v{i}]xfade=transition=fade:duration={transition_duration}:offset={offset}[xf{i-1}]'
                )
            
            # Last xfade
            last_idx = len(image_paths) - 1
            offset = offset + duration - transition_duration
            filter_parts.append(
                f'[xf{last_idx-2}][v{last_idx}]xfade=transition=fade:duration={transition_duration}:offset={offset}[outv]'
            )
        
        filter_complex = ';'.join(filter_parts)
        
        cmd = [
            'ffmpeg', '-y',
            *inputs,
            '-filter_complex', filter_complex,
            '-map', '[outv]',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'fast',
            '-crf', '23',
            '-movflags', '+faststart',
            output_path
        ]
        
        logger.info(f"Running FFmpeg with fade transitions")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg xfade error: {result.stderr}")
            # Fallback to simple slideshow
            logger.info("Falling back to simple slideshow")
            return create_simple_slideshow(image_paths, output_path, duration, fps, width, height)
        
        return os.path.exists(output_path)
        
    except Exception as e:
        logger.error(f"Fade slideshow error: {e}")
        return create_simple_slideshow(image_paths, output_path, duration, fps, width, height)


def create_slide_slideshow(
    image_paths: List[str],
    output_path: str,
    duration: float,
    transition_duration: float,
    fps: int,
    width: int,
    height: int
) -> bool:
    """Create slideshow with slide transitions"""
    
    if len(image_paths) < 2:
        return create_simple_slideshow(image_paths, output_path, duration, fps, width, height)
    
    try:
        inputs = []
        for path in image_paths:
            inputs.extend(['-loop', '1', '-t', str(duration + transition_duration), '-i', path])
        
        filter_parts = []
        
        # Scale all inputs
        for i in range(len(image_paths)):
            filter_parts.append(
                f'[{i}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,'
                f'pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps={fps}[v{i}]'
            )
        
        # Alternate slide directions
        directions = ['slideleft', 'slideright', 'slideup', 'slidedown']
        
        if len(image_paths) == 2:
            offset = duration - transition_duration
            filter_parts.append(
                f'[v0][v1]xfade=transition=slideleft:duration={transition_duration}:offset={offset}[outv]'
            )
        else:
            offset = duration - transition_duration
            filter_parts.append(
                f'[v0][v1]xfade=transition={directions[0]}:duration={transition_duration}:offset={offset}[xf0]'
            )
            
            for i in range(2, len(image_paths) - 1):
                prev_offset = offset
                offset = prev_offset + duration - transition_duration
                direction = directions[i % len(directions)]
                filter_parts.append(
                    f'[xf{i-2}][v{i}]xfade=transition={direction}:duration={transition_duration}:offset={offset}[xf{i-1}]'
                )
            
            last_idx = len(image_paths) - 1
            offset = offset + duration - transition_duration
            direction = directions[last_idx % len(directions)]
            filter_parts.append(
                f'[xf{last_idx-2}][v{last_idx}]xfade=transition={direction}:duration={transition_duration}:offset={offset}[outv]'
            )
        
        filter_complex = ';'.join(filter_parts)
        
        cmd = [
            'ffmpeg', '-y',
            *inputs,
            '-filter_complex', filter_complex,
            '-map', '[outv]',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'fast',
            '-crf', '23',
            '-movflags', '+faststart',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg slide error: {result.stderr}")
            return create_simple_slideshow(image_paths, output_path, duration, fps, width, height)
        
        return os.path.exists(output_path)
        
    except Exception as e:
        logger.error(f"Slide slideshow error: {e}")
        return create_simple_slideshow(image_paths, output_path, duration, fps, width, height)


def create_zoom_slideshow(
    image_paths: List[str],
    output_path: str,
    duration: float,
    fps: int,
    width: int,
    height: int
) -> bool:
    """Create slideshow with Ken Burns zoom effect"""
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        segment_paths = []
        
        # Create zoompan segments for each image
        for i, img_path in enumerate(image_paths):
            segment_path = os.path.join(temp_dir, f'segment_{i}.mp4')
            
            # Alternate between zoom in and zoom out
            if i % 2 == 0:
                zoom_filter = f"zoompan=z='min(zoom+0.001,1.2)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={int(duration*fps)}:s={width}x{height}:fps={fps}"
            else:
                zoom_filter = f"zoompan=z='if(lte(zoom,1.0),1.2,max(1.001,zoom-0.001))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={int(duration*fps)}:s={width}x{height}:fps={fps}"
            
            cmd = [
                'ffmpeg', '-y',
                '-loop', '1',
                '-i', img_path,
                '-vf', f'scale=8000:-1,{zoom_filter}',
                '-t', str(duration),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-preset', 'fast',
                segment_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                segment_paths.append(segment_path)
            else:
                logger.warning(f"Zoom segment {i} failed, using static")
                # Create static segment as fallback
                static_cmd = [
                    'ffmpeg', '-y',
                    '-loop', '1',
                    '-i', img_path,
                    '-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black',
                    '-t', str(duration),
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    segment_path
                ]
                subprocess.run(static_cmd, capture_output=True, timeout=60)
                if os.path.exists(segment_path):
                    segment_paths.append(segment_path)
        
        if not segment_paths:
            return create_simple_slideshow(image_paths, output_path, duration, fps, width, height)
        
        # Concatenate segments
        concat_file = os.path.join(temp_dir, 'concat.txt')
        with open(concat_file, 'w') as f:
            for seg_path in segment_paths:
                f.write(f"file '{seg_path}'\n")
        
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        return result.returncode == 0 and os.path.exists(output_path)
        
    except Exception as e:
        logger.error(f"Zoom slideshow error: {e}")
        return create_simple_slideshow(image_paths, output_path, duration, fps, width, height)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.route('/health', methods=['GET'])
@app.route('/', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy", 
        "service": "frame-extractor-with-branding", 
        "version": "4.0.0",
        "endpoints": [
            "/extract-frames",
            "/extract-all-frames",
            "/brand-image",
            "/create-slideshow",
            "/videos/<video_id>.mp4"
        ]
    }), 200


@app.route('/brand-image', methods=['POST'])
def brand_image():
    """Apply logo/branding overlay to an image"""
    try:
        if request.json is None:
            return jsonify({"success": False, "error": "No JSON data"}), 400
        
        data = request.json
        
        # Get base image (from base64 or URL)
        if 'image_base64' in data and data['image_base64']:
            image_data = base64.b64decode(data['image_base64'])
            base_image = Image.open(io.BytesIO(image_data)).convert('RGBA')
            logger.info("Loaded base image from base64")
        elif 'image_url' in data and data['image_url']:
            base_image = download_image_from_url(data['image_url'])
            logger.info("Loaded base image from URL")
        else:
            return jsonify({"success": False, "error": "No image provided (image_base64 or image_url required)"}), 400
        
        # Get logo (from Telegram file_id or URL)
        logo = None
        if 'logo_file_id' in data and data['logo_file_id']:
            bot_token = data.get('bot_token')
            if not bot_token:
                return jsonify({"success": False, "error": "bot_token required when using logo_file_id"}), 400
            logo = download_image_from_telegram(data['logo_file_id'], bot_token)
            logger.info("Loaded logo from Telegram")
        elif 'logo_url' in data and data['logo_url']:
            logo = download_image_from_url(data['logo_url'])
            logger.info("Loaded logo from URL")
        elif 'logo_base64' in data and data['logo_base64']:
            logo_data = base64.b64decode(data['logo_base64'])
            logo = Image.open(io.BytesIO(logo_data)).convert('RGBA')
            logger.info("Loaded logo from base64")
        
        # If no logo, just return the original image
        if logo is None:
            logger.info("No logo provided, returning original image")
            # Convert to RGB for JPEG output
            if base_image.mode == 'RGBA':
                background = Image.new('RGB', base_image.size, (255, 255, 255))
                background.paste(base_image, mask=base_image.split()[3])
                base_image = background
            
            buffer = io.BytesIO()
            base_image.save(buffer, format='JPEG', quality=95)
            buffer.seek(0)
            
            return jsonify({
                "success": True,
                "image_base64": base64.b64encode(buffer.getvalue()).decode('utf-8'),
                "branded": False
            })
        
        # Get branding parameters
        position = data.get('position', 'bottom-right')
        opacity = float(data.get('opacity', 0.8))
        size_percent = float(data.get('size', 12))
        
        logger.info(f"Applying logo: position={position}, opacity={opacity}, size={size_percent}%")
        
        # Apply logo overlay
        branded_image = apply_logo_overlay(
            base_image=base_image,
            logo=logo,
            position=position,
            opacity=opacity,
            size_percent=size_percent
        )
        
        # Convert RGBA to RGB for JPEG output
        if branded_image.mode == 'RGBA':
            background = Image.new('RGB', branded_image.size, (255, 255, 255))
            background.paste(branded_image, mask=branded_image.split()[3])
            branded_image = background
        
        # Save to buffer
        buffer = io.BytesIO()
        branded_image.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)
        
        logger.info("✅ Branding applied successfully")
        
        return jsonify({
            "success": True,
            "image_base64": base64.b64encode(buffer.getvalue()).decode('utf-8'),
            "branded": True,
            "width": branded_image.width,
            "height": branded_image.height
        })
        
    except Exception as e:
        logger.error(f"Branding error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/create-slideshow', methods=['POST'])
def create_slideshow():
    """Create a slideshow video from multiple images"""
    temp_dir = None
    
    try:
        if request.json is None:
            return jsonify({"success": False, "error": "No JSON data"}), 400
        
        data = request.json
        image_urls = data.get('image_urls', [])
        
        if not image_urls or len(image_urls) < 1:
            return jsonify({"success": False, "error": "At least 1 image URL required"}), 400
        
        # Get parameters
        duration_per_image = float(data.get('duration_per_image', 3.0))
        transition = data.get('transition', 'fade')  # fade, slide, zoom, none
        transition_duration = float(data.get('transition_duration', 0.5))
        fps = int(data.get('fps', 30))
        aspect_ratio = data.get('aspect_ratio', '9:16')
        
        logger.info(f"Creating slideshow: {len(image_urls)} images, {duration_per_image}s each, transition={transition}")
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        
        # Download all images
        image_paths = []
        for i, url in enumerate(image_urls):
            img_path = os.path.join(temp_dir, f'img_{i:03d}.jpg')
            if download_image_to_file(url, img_path):
                image_paths.append(img_path)
                logger.info(f"Downloaded image {i+1}/{len(image_urls)}")
            else:
                logger.warning(f"Failed to download image {i+1}: {url}")
        
        if len(image_paths) < 1:
            return jsonify({"success": False, "error": "Failed to download any images"}), 400
        
        logger.info(f"Successfully downloaded {len(image_paths)} images")
        
        # Generate video ID and output path
        video_id = str(uuid4())
        output_filename = f'{video_id}.mp4'
        output_path = os.path.join(VIDEO_STORAGE_DIR, output_filename)
        
        # Create slideshow
        success = create_slideshow_with_ffmpeg(
            image_paths=image_paths,
            output_path=output_path,
            duration_per_image=duration_per_image,
            transition=transition,
            transition_duration=transition_duration,
            fps=fps,
            aspect_ratio=aspect_ratio
        )
        
        if not success or not os.path.exists(output_path):
            return jsonify({"success": False, "error": "Video creation failed"}), 500
        
        # Get video file size
        file_size = os.path.getsize(output_path)
        
        # Calculate total duration
        if transition != 'none' and len(image_paths) > 1:
            total_duration = (len(image_paths) * duration_per_image) - ((len(image_paths) - 1) * transition_duration)
        else:
            total_duration = len(image_paths) * duration_per_image
        
        # Get the base URL from request
        base_url = request.host_url.rstrip('/')
        video_url = f"{base_url}/videos/{output_filename}"
        
        logger.info(f"✅ Slideshow created: {output_filename}, {file_size} bytes, {total_duration:.1f}s")
        
        return jsonify({
            "success": True,
            "video_url": video_url,
            "video_id": video_id,
            "duration": round(total_duration, 2),
            "scenes": len(image_paths),
            "file_size": file_size,
            "aspect_ratio": aspect_ratio,
            "transition": transition
        })
        
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg timeout")
        return jsonify({"success": False, "error": "Video creation timed out"}), 500
    except Exception as e:
        logger.error(f"Slideshow error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.route('/videos/<video_id>', methods=['GET'])
def serve_video(video_id):
    """Serve generated video files"""
    try:
        # Sanitize video_id to prevent directory traversal
        if '..' in video_id or '/' in video_id:
            return jsonify({"error": "Invalid video ID"}), 400
        
        video_path = os.path.join(VIDEO_STORAGE_DIR, video_id)
        
        if not os.path.exists(video_path):
            return jsonify({"error": "Video not found"}), 404
        
        return send_file(
            video_path,
            mimetype='video/mp4',
            as_attachment=False,
            download_name=video_id
        )
        
    except Exception as e:
        logger.error(f"Video serve error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/extract-frames', methods=['POST'])
def extract_frames():
    """Extract specific frames from video at given timestamps"""
    video_path: Optional[str] = None
    cap = None
    try:
        if request.json is None:
            return jsonify({"success": False, "error": "No JSON data"}), 400
        
        data = request.json
        video_url = data.get('video_url')
        timestamps = data.get('timestamps', [])
        
        if not video_url or not timestamps:
            return jsonify({"success": False, "error": "video_url and timestamps required"}), 400
        
        logger.info("Downloading video")
        response = requests.get(str(video_url), stream=True, timeout=60)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            for chunk in response.iter_content(8192):
                tmp.write(chunk)
            video_path = tmp.name
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({"success": False, "error": "Failed to open video"}), 500
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {total/fps:.1f}s, {fps:.1f}fps")
        
        frames: List[Dict[str, Any]] = []
        skipped = 0
        
        for ts in timestamps:
            frame_num = int(ts * fps)
            if frame_num >= total:
                continue
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret or not should_keep_frame(frame):
                if not ret:
                    logger.warning(f"Failed to read frame at {ts}s")
                else:
                    skipped += 1
                continue
            
            h, w = frame.shape[:2]
            if w > 1280:
                frame = cv2.resize(frame, (1280, int(h * 1280 / w)))
            
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frames.append({
                "timestamp": ts,
                "image_base64": base64.b64encode(buf).decode(),
                "frame_number": frame_num
            })
        
        logger.info(f"✅ {len(frames)} with faces, {skipped} skipped")
        return jsonify({"success": True, "frames": frames, "count": len(frames), "skipped": skipped}), 200
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if cap:
            cap.release()
        if video_path and os.path.exists(video_path):
            os.unlink(video_path)


@app.route('/extract-all-frames', methods=['POST'])
def extract_all_frames():
    """Extract all frames with faces from video"""
    video_path: Optional[str] = None
    cap = None
    try:
        if request.json is None:
            return jsonify({"success": False, "error": "No JSON data"}), 400
        
        data = request.json
        video_url = data.get('video_url')
        if not video_url:
            return jsonify({"success": False, "error": "video_url required"}), 400
        
        logger.info("Downloading video")
        response = requests.get(str(video_url), stream=True, timeout=60)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            for chunk in response.iter_content(8192):
                tmp.write(chunk)
            video_path = tmp.name
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({"success": False, "error": "Failed to open video"}), 500
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total / fps if fps > 0 else 0
        
        logger.info(f"Video: {duration:.1f}s, {fps:.1f}fps, {total} frames")
        
        frames: List[Dict[str, Any]] = []
        skipped = 0
        
        for idx in range(0, total, 10):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            if not should_keep_frame(frame):
                skipped += 1
                continue
            
            h, w = frame.shape[:2]
            if w > 1280:
                frame = cv2.resize(frame, (1280, int(h * 1280 / w)))
            
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frames.append({
                "timestamp": round(idx / fps, 2),
                "image_base64": base64.b64encode(buf).decode()
            })
            
            if len(frames) % 30 == 0 and len(frames) > 0:
                logger.info(f"{len(frames)} with faces, {skipped} skipped")
        
        logger.info(f"✅ Final: {len(frames)} with faces, {skipped} skipped")
        return jsonify({"success": True, "frames": frames, "count": len(frames), "skipped": skipped, "duration": round(duration, 2)}), 200
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if cap:
            cap.release()
        if video_path and os.path.exists(video_path):
            os.unlink(video_path)


@app.route('/extract-reel-frames', methods=['POST'])
def extract_reel_frames():
    """Extract smart frames from Instagram Reel URL for carousel"""
    video_path: Optional[str] = None
    cap = None
    
    try:
        if request.json is None:
            return jsonify({"success": False, "error": "No JSON data"}), 400
        
        data = request.json
        reel_url = data.get('reel_url')
        frame_count = int(data.get('frame_count', 10))
        smart_selection = data.get('smart_selection', True)
        
        if not reel_url:
            return jsonify({"success": False, "error": "reel_url required"}), 400
        
        logger.info(f"Processing reel: {reel_url}")
        
        # Download reel using yt-dlp
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            video_path = tmp.name
        
        # Use yt-dlp to download Instagram reel
        cmd = [
            'yt-dlp',
            '--no-check-certificate',
            '-f', 'best[ext=mp4]/best',
            '-o', video_path,
            '--no-playlist',
            reel_url
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0 or not os.path.exists(video_path):
            logger.error(f"yt-dlp error: {result.stderr}")
            return jsonify({"success": False, "error": "Failed to download reel. Make sure the URL is valid and public."}), 400
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({"success": False, "error": "Failed to open video"}), 500
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total / fps if fps > 0 else 0
        
        logger.info(f"Reel: {duration:.1f}s, {fps:.1f}fps, {total} frames")
        
        # Calculate frame interval
        interval = max(1, total // (frame_count * 2))  # Sample more frames for smart selection
        
        candidate_frames: List[Dict[str, Any]] = []
        
        for idx in range(0, total, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate frame score based on various factors
            score = 0
            
            # Check for faces (higher score)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) > 0:
                score += 50
                # Bigger faces = higher score
                for (x, y, w, h) in faces:
                    score += (w * h) / (frame.shape[0] * frame.shape[1]) * 100
            
            # Calculate image sharpness (Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            score += min(sharpness / 100, 30)  # Cap sharpness contribution
            
            # Calculate color variance (more colorful = higher score)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            color_var = hsv[:, :, 1].std()
            score += min(color_var / 10, 20)
            
            h, w = frame.shape[:2]
            if w > 1280:
                frame = cv2.resize(frame, (1280, int(h * 1280 / w)))
            
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            candidate_frames.append({
                "timestamp": round(idx / fps, 2),
                "frame_number": idx,
                "score": score,
                "image_base64": base64.b64encode(buf).decode()
            })
        
        # Smart selection: pick best frames with good distribution
        if smart_selection and len(candidate_frames) > frame_count:
            # Sort by score
            candidate_frames.sort(key=lambda x: x['score'], reverse=True)
            
            # Select top frames ensuring good time distribution
            selected_frames = []
            time_buckets = [False] * frame_count
            bucket_duration = duration / frame_count
            
            for frame in candidate_frames:
                bucket_idx = min(int(frame['timestamp'] / bucket_duration), frame_count - 1)
                if not time_buckets[bucket_idx] and len(selected_frames) < frame_count:
                    selected_frames.append(frame)
                    time_buckets[bucket_idx] = True
            
            # Fill remaining slots with highest scored frames
            for frame in candidate_frames:
                if len(selected_frames) >= frame_count:
                    break
                if frame not in selected_frames:
                    selected_frames.append(frame)
            
            # Sort by timestamp for proper order
            selected_frames.sort(key=lambda x: x['timestamp'])
            frames = selected_frames
        else:
            # Simple selection: evenly spaced
            step = max(1, len(candidate_frames) // frame_count)
            frames = candidate_frames[::step][:frame_count]
        
        # Remove score from output
        for frame in frames:
            frame.pop('score', None)
        
        logger.info(f"✅ Extracted {len(frames)} smart frames from reel")
        
        return jsonify({
            "success": True,
            "frames": frames,
            "count": len(frames),
            "duration": round(duration, 2),
            "source": "instagram_reel"
        })
        
    except subprocess.TimeoutExpired:
        logger.error("Download timeout")
        return jsonify({"success": False, "error": "Reel download timed out"}), 500
    except Exception as e:
        logger.error(f"Reel extraction error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if cap:
            cap.release()
        if video_path and os.path.exists(video_path):
            os.unlink(video_path)


@app.route('/add-audio-to-video', methods=['POST'])
def add_audio_to_video():
    """Add audio track to a video"""
    temp_dir = None
    
    try:
        if request.json is None:
            return jsonify({"success": False, "error": "No JSON data"}), 400
        
        data = request.json
        video_url = data.get('video_url')
        audio_url = data.get('audio_url')
        
        if not video_url or not audio_url:
            return jsonify({"success": False, "error": "video_url and audio_url required"}), 400
        
        temp_dir = tempfile.mkdtemp()
        
        # Download video
        video_path = os.path.join(temp_dir, 'video.mp4')
        response = requests.get(video_url, timeout=60)
        response.raise_for_status()
        with open(video_path, 'wb') as f:
            f.write(response.content)
        
        # Download audio
        audio_path = os.path.join(temp_dir, 'audio.mp3')
        response = requests.get(audio_url, timeout=60)
        response.raise_for_status()
        with open(audio_path, 'wb') as f:
            f.write(response.content)
        
        # Generate output
        video_id = str(uuid4())
        output_path = os.path.join(VIDEO_STORAGE_DIR, f'{video_id}.mp4')
        
        # Get video duration
        probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
        duration_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        video_duration = float(duration_result.stdout.strip()) if duration_result.returncode == 0 else 30
        
        # Merge video and audio
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-shortest',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-t', str(video_duration),
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0 or not os.path.exists(output_path):
            return jsonify({"success": False, "error": "Failed to add audio"}), 500
        
        base_url = request.host_url.rstrip('/')
        video_url_out = f"{base_url}/videos/{video_id}.mp4"
        
        return jsonify({
            "success": True,
            "video_url": video_url_out,
            "video_id": video_id,
            "duration": round(video_duration, 2)
        })
        
    except Exception as e:
        logger.error(f"Add audio error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.errorhandler(413)
def too_large(error):
    return jsonify({"success": False, "error": "File too large (max 100MB)"}), 413


@app.errorhandler(500)
def server_error(error):
    return jsonify({"success": False, "error": "Internal error"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)), debug=False)