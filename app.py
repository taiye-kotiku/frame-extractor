from flask import Flask, request, jsonify, send_file
from PIL import Image
import cv2
import base64
import requests
import tempfile
import os
import io
import logging
from typing import Optional, Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # type: ignore


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


@app.route('/health', methods=['GET'])
@app.route('/', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy", 
        "service": "frame-extractor-with-branding", 
        "version": "3.0.0",
        "endpoints": [
            "/extract-frames",
            "/extract-all-frames",
            "/brand-image"
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


@app.route('/extract-frames', methods=['POST'])
def extract_frames():
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


@app.errorhandler(413)
def too_large(error):
    return jsonify({"success": False, "error": "File too large (max 100MB)"}), 413


@app.errorhandler(500)
def server_error(error):
    return jsonify({"success": False, "error": "Internal error"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)), debug=False)