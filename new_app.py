from flask import Flask, request, jsonify
import cv2
import base64
import requests
import tempfile
import os
import logging
from typing import Optional, Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Increase max content length for video uploads
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def should_keep_frame(frame):
    """Checks if a face is detected and is at least 1/6th of image height or width."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    img_h, img_w = frame.shape[:2]
    for (x, y, w, h) in faces:
        if w >= (img_w / 6) or h >= (img_h / 6):
            return True
    return False

@app.route('/health', methods=['GET'])
@app.route('/', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "service": "frame-extractor",
        "version": "2.1.0"
    }), 200

@app.route('/extract-frames', methods=['POST'])
def extract_frames():
    video_path: Optional[str] = None
    try:
        if request.json is None:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
        
        data: Dict[str, Any] = request.json
        video_url: Optional[str] = data.get('video_url')
        timestamps: List[float] = data.get('timestamps', [])
        
        if not video_url or not timestamps:
            return jsonify({"success": False, "error": "video_url and timestamps required"}), 400
        
        response = requests.get(video_url, stream=True, timeout=60)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            video_path = tmp_file.name
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames: List[Dict[str, Any]] = []
        
        for timestamp in timestamps:
            frame_number = int(timestamp * fps)
            if frame_number >= total_frames:
                continue
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret and should_keep_frame(frame): # Added face check
                # Resize if needed
                height, width = frame.shape[:2]
                if width > 1280:
                    scale = 1280 / width
                    frame = cv2.resize(frame, (1280, int(height * scale)))
                
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
                frames.append({
                    "timestamp": timestamp,
                    "image_base64": frame_b64,
                    "frame_number": frame_number
                })
        
        cap.release()
        return jsonify({"success": True, "frames": frames, "count": len(frames)}), 200
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if video_path and os.path.exists(video_path):
            os.unlink(video_path)

@app.route('/extract-all-frames', methods=['POST'])
def extract_all_frames():
    video_path: Optional[str] = None
    cap = None
    try:
        data: Dict[str, Any] = request.json
        video_url = data.get('video_url')
        
        response = requests.get(video_url, stream=True, timeout=60)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            video_path = tmp_file.name
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        frames: List[Dict[str, Any]] = []
        # Modified to process every 10th frame globally for efficiency
        for f_idx in range(0, total_frames, 10):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
                
            if should_keep_frame(frame): # Added face check
                height, width = frame.shape[:2]
                if width > 1280:
                    scale = 1280 / width
                    frame = cv2.resize(frame, (1280, int(height * scale)))
                
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
                frames.append({
                    "timestamp": round(f_idx / fps, 2),
                    "image_base64": frame_b64
                })

        return jsonify({
            "success": True, 
            "frames": frames, 
            "count": len(frames),
            "duration": round(duration, 2)
        }), 200
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if cap: cap.release()
        if video_path and os.path.exists(video_path):
            os.unlink(video_path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)