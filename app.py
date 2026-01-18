from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import requests
from io import BytesIO
import tempfile
import os

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "frame-extractor"}), 200

@app.route('/extract-frames', methods=['POST'])
def extract_frames():
    """
    Extract frames from video at specific timestamps
    
    Request JSON:
    {
        "video_url": "https://instagram-video.mp4",
        "timestamps": [2.5, 7.0, 12.0, 18.0, 23.0, 28.0, 35.0, 40.0, 45.0, 50.0],
        "max_frames": 50  # optional: extract all frames for AI analysis
    }
    
    Response JSON:
    {
        "success": true,
        "frames": [
            {
                "timestamp": 2.5,
                "image_base64": "...",
                "frame_number": 75
            },
            ...
        ],
        "metadata": {
            "duration": 60.5,
            "fps": 30,
            "total_frames": 1815,
            "resolution": "1080x1920"
        }
    }
    """
    try:
        data = request.json
        video_url = data.get('video_url')
        timestamps = data.get('timestamps', [])
        max_frames = data.get('max_frames')  # For "extract all" mode
        
        if not video_url:
            return jsonify({"success": False, "error": "video_url required"}), 400
        
        # Download video
        print(f"Downloading video from {video_url}")
        response = requests.get(video_url, stream=True, timeout=60)
        response.raise_for_status()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            video_path = tmp_file.name
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            os.unlink(video_path)
            return jsonify({"success": False, "error": "Failed to open video"}), 500
        
        # Get video metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frames = []
        
        if max_frames:
            # Extract evenly spaced frames (for AI analysis)
            frame_interval = max(1, total_frames // max_frames)
            
            for i in range(0, total_frames, frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Encode frame to JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
                frames.append({
                    "timestamp": round(i / fps, 2),
                    "image_base64": frame_b64,
                    "frame_number": i
                })
                
                if len(frames) >= max_frames:
                    break
        
        else:
            # Extract frames at specific timestamps
            for timestamp in timestamps:
                # Calculate frame number
                frame_number = int(timestamp * fps)
                
                if frame_number >= total_frames:
                    continue
                
                # Seek to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Encode frame to JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
                frames.append({
                    "timestamp": timestamp,
                    "image_base64": frame_b64,
                    "frame_number": frame_number
                })
        
        # Cleanup
        cap.release()
        os.unlink(video_path)
        
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
        
    except requests.exceptions.RequestException as e:
        return jsonify({"success": False, "error": f"Video download failed: {str(e)}"}), 500
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/extract-all-frames', methods=['POST'])
def extract_all_frames():
    """
    Extract ALL frames from video (1 per second for AI analysis)
    
    Request JSON:
    {
        "video_url": "https://instagram-video.mp4"
    }
    
    Response JSON:
    {
        "success": true,
        "frames": [...],  # All frames at 1fps
        "count": 60
    }
    """
    try:
        data = request.json
        video_url = data.get('video_url')
        
        if not video_url:
            return jsonify({"success": False, "error": "video_url required"}), 400
        
        # Download video
        response = requests.get(video_url, stream=True, timeout=60)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            video_path = tmp_file.name
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            os.unlink(video_path)
            return jsonify({"success": False, "error": "Failed to open video"}), 500
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Extract 1 frame per second
        frames = []
        for second in range(int(duration)):
            frame_number = int(second * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
                frames.append({
                    "timestamp": second,
                    "image_base64": frame_b64,
                    "frame_number": frame_number
                })
        
        cap.release()
        os.unlink(video_path)
        
        return jsonify({
            "success": True,
            "frames": frames,
            "count": len(frames),
            "duration": round(duration, 2)
        }), 200
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)