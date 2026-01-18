from flask import Flask, request, jsonify, Request
import cv2
import numpy as np
import base64
import requests
from io import BytesIO
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

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "frame-extractor",
        "version": "1.0.0"
    }), 200

@app.route('/extract-frames', methods=['POST'])
def extract_frames():
    """
    Extract frames from video at specific timestamps
    
    Request JSON:
    {
        "video_url": "https://instagram-video.mp4",
        "timestamps": [2.5, 7.0, 12.0, 18.0, 23.0, 28.0, 35.0, 40.0, 45.0, 50.0]
    }
    
    Response JSON:
    {
        "success": true,
        "frames": [
            {
                "timestamp": 2.5,
                "image_base64": "...",
                "frame_number": 75
            }
        ],
        "metadata": {
            "duration": 60.5,
            "fps": 30,
            "total_frames": 1815,
            "resolution": "1080x1920"
        }
    }
    """
    video_path: Optional[str] = None
    
    try:
        # Type-safe request.json access
        if request.json is None:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
        
        data: Dict[str, Any] = request.json
        video_url: Optional[str] = data.get('video_url')
        timestamps: List[float] = data.get('timestamps', [])
        
        if not video_url:
            return jsonify({"success": False, "error": "video_url required"}), 400
        
        if not timestamps:
            return jsonify({"success": False, "error": "timestamps required"}), 400
        
        logger.info(f"Downloading video from {video_url}")
        
        # Download video with timeout
        response = requests.get(video_url, stream=True, timeout=60)
        response.raise_for_status()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            video_path = tmp_file.name
        
        logger.info(f"Video downloaded to {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return jsonify({"success": False, "error": "Failed to open video"}), 500
        
        # Get video metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video metadata: {duration}s, {fps}fps, {total_frames} frames, {width}x{height}")
        
        frames: List[Dict[str, Any]] = []
        
        # Extract frames at specific timestamps
        for timestamp in timestamps:
            # Calculate frame number
            frame_number = int(timestamp * fps)
            
            if frame_number >= total_frames:
                logger.warning(f"Timestamp {timestamp}s exceeds video duration")
                continue
            
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if not ret:
                logger.warning(f"Failed to read frame at {timestamp}s")
                continue
            
            # Encode frame to JPEG with quality 85
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
            _, buffer = cv2.imencode('.jpg', frame, encode_params)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            frames.append({
                "timestamp": timestamp,
                "image_base64": frame_b64,
                "frame_number": frame_number
            })
            
            logger.info(f"Extracted frame at {timestamp}s")
        
        # Cleanup
        cap.release()
        
        logger.info(f"Successfully extracted {len(frames)} frames")
        
        return jsonify({
            "success": True,
            "frames": frames,
            "metadata": {
                "duration": round(duration, 2),
                "fps": round(fps, 2),
                "total_frames": total_frames,
                "resolution": f"{width}x{height}"
            }
        }), 200
        
    except requests.exceptions.Timeout:
        logger.error("Video download timeout")
        return jsonify({"success": False, "error": "Video download timeout"}), 504
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Video download failed: {str(e)}")
        return jsonify({"success": False, "error": f"Video download failed: {str(e)}"}), 500
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
    
    finally:
        # Always cleanup temp file
        if video_path and os.path.exists(video_path):
            try:
                os.unlink(video_path)
                logger.info(f"Cleaned up temp file: {video_path}")
            except Exception as e:
                logger.error(f"Failed to cleanup temp file: {str(e)}")

@app.route('/extract-all-frames', methods=['POST'])
def extract_all_frames():
    """
    Extract ALL frames from video (1 per second for AI analysis)
    
    Request JSON:
    {
        "video_url": "https://instagram-video.mp4",
        "max_duration": 300  # optional: max 5 minutes
    }
    
    Response JSON:
    {
        "success": true,
        "frames": [...],
        "count": 60,
        "duration": 60.5
    }
    """
    video_path: Optional[str] = None
    
    try:
        # Type-safe request.json access
        if request.json is None:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
        
        data: Dict[str, Any] = request.json
        video_url: Optional[str] = data.get('video_url')
        max_duration: int = data.get('max_duration', 300)  # 5 minutes default
        
        if not video_url:
            return jsonify({"success": False, "error": "video_url required"}), 400
        
        logger.info(f"Downloading video from {video_url}")
        
        # Download video
        response = requests.get(video_url, stream=True, timeout=60)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            video_path = tmp_file.name
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return jsonify({"success": False, "error": "Failed to open video"}), 500
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video: {duration}s, {fps}fps, {total_frames} frames")
        
        # Check max duration
        if duration > max_duration:
            cap.release()
            return jsonify({
                "success": False,
                "error": f"Video too long ({duration}s > {max_duration}s max)"
            }), 400
        
        # Extract 1 frame per second
        frames: List[Dict[str, Any]] = []
        for second in range(int(duration)):
            frame_number = int(second * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                # Encode with slightly lower quality for faster processing
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]
                _, buffer = cv2.imencode('.jpg', frame, encode_params)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
                frames.append({
                    "timestamp": second,
                    "image_base64": frame_b64,
                    "frame_number": frame_number
                })
                
                logger.info(f"Extracted frame at {second}s")
        
        cap.release()
        
        logger.info(f"Successfully extracted {len(frames)} frames")
        
        return jsonify({
            "success": True,
            "frames": frames,
            "count": len(frames),
            "duration": round(duration, 2)
        }), 200
        
    except requests.exceptions.Timeout:
        logger.error("Video download timeout")
        return jsonify({"success": False, "error": "Video download timeout"}), 504
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Video download failed: {str(e)}")
        return jsonify({"success": False, "error": f"Video download failed: {str(e)}"}), 500
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
    
    finally:
        if video_path and os.path.exists(video_path):
            try:
                os.unlink(video_path)
                logger.info(f"Cleaned up temp file: {video_path}")
            except Exception as e:
                logger.error(f"Failed to cleanup temp file: {str(e)}")

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "success": False,
        "error": "Video file too large (max 100MB)"
    }), 413

@app.errorhandler(500)
def internal_server_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)