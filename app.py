from flask import Flask, request, jsonify
import cv2
import base64
import requests
import tempfile
import os
import logging
from typing import Optional, Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# Load Haar Cascade for face detection (fixes Pylance cv2.data error)
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
        return True  # Keep frame if error

@app.route('/health', methods=['GET'])
@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "service": "frame-extractor-face-detection", "version": "2.1.0"}), 200

@app.route('/extract-frames', methods=['POST'])
def extract_frames():
    video_path: Optional[str] = None
    cap = None
    try:
        if request.json is None:
            return jsonify({"success": False, "error": "No JSON data"}), 400
        
        # Fixes Pylance errors - let Python infer types
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
        
        # Every 10th frame = ~3fps at 30fps
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
    return jsonify({"success": False, "error": "Video too large (max 100MB)"}), 413

@app.errorhandler(500)
def server_error(error):
    return jsonify({"success": False, "error": "Internal error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)), debug=False)