from flask import Flask, render_template, request, jsonify, send_from_directory, Response, url_for
import os
from PIL import Image
import io
import base64
from ultralytics import YOLO
import cv2
import numpy as np
import threading
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained YOLO model
model = YOLO('runs/train/exp4/weights/best.pt')

# Global variables for webcam
camera = None
output_frame = None
lock = threading.Lock()
webcam_active = False
detection_count = 0
webcam_error = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        try:
            # Create a safe filename to prevent path traversal
            import uuid
            from werkzeug.utils import secure_filename
            
            # Get file extension
            ext = os.path.splitext(file.filename)[1].lower()
            # Create a safe filename using UUID
            safe_filename = secure_filename(f"{uuid.uuid4().hex}{ext}")
            
            # Save the uploaded image
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
            file.save(img_path)
            
            # Perform detection
            results = model(img_path)
            
            # Process results
            result = results[0]
            
            # Read the image with OpenCV
            img = cv2.imread(img_path)
            if img is None:
                return jsonify({'error': 'Could not read uploaded image'})
            
            # Get bounding boxes, classes and confidence scores
            boxes = result.boxes
            
            # Draw bounding boxes on the image
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add confidence score
                conf = box.conf[0]
                cv2.putText(img, f'ID Card: {conf:.2f}', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Save the annotated image
            result_filename = f"result_{safe_filename}"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            cv2.imwrite(output_path, img)
            
            # Return the result
            return jsonify({
                'success': True,
                'original': safe_filename,
                'result': result_filename,
                'num_detections': len(boxes)
            })
        except Exception as e:
            print(f"Error processing image: {e}")
            return jsonify({'error': f'Error processing image: {str(e)}'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    try:
        # Ensure the file exists
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            return "File not found", 404
            
        # Log file access attempt
        print(f"Serving file: {file_path}")
        
        # Determine file type
        mime_type = 'image/jpeg'
        if filename.lower().endswith('.png'):
            mime_type = 'image/png'
        
        # Serve the file with appropriate MIME type and cache control headers
        response = send_from_directory(
            app.config['UPLOAD_FOLDER'], 
            filename, 
            mimetype=mime_type
        )
        
        # Add cache control headers to prevent caching
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        
        return response
    except Exception as e:
        print(f"Error serving file {filename}: {e}")
        return f"Error: {str(e)}", 500

def detect_id_cards_webcam(frame):
    global detection_count
    
    # Perform detection using the model
    results = model(frame)
    result = results[0]
    
    # Get bounding boxes
    boxes = result.boxes
    
    # Update the global detection count
    detection_count = len(boxes)
    
    # Draw bounding boxes on the frame
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add confidence score
        conf = box.conf[0]
        cv2.putText(frame, f'ID Card: {conf:.2f}', (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                   
    return frame, len(boxes)

def webcam_stream():
    global camera, output_frame, lock, webcam_active, webcam_error
    
    try:
        # Try different camera indices if the default doesn't work
        for camera_index in range(2):
            print(f"Trying camera index {camera_index}")
            camera = cv2.VideoCapture(camera_index)
            if camera.isOpened():
                print(f"Successfully opened camera {camera_index}")
                break
            else:
                print(f"Failed to open camera {camera_index}")
                camera.release()
        
        if not camera.isOpened():
            webcam_error = "Could not open webcam. Please check your camera connection."
            print(webcam_error)
            webcam_active = False
            return
            
        # Initialize the camera with specific parameters
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        time.sleep(2.0)  # Allow camera to warm up
        
        # Get a test frame
        success, test_frame = camera.read()
        if not success or test_frame is None:
            webcam_error = "Could not read from webcam. Please check your camera settings."
            print(webcam_error)
            webcam_active = False
            camera.release()
            return
            
        print("Webcam initialized successfully, starting detection loop")
        
        # Create a placeholder frame with a message
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Starting webcam...", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        with lock:
            output_frame = placeholder.copy()
        
        while webcam_active:
            success, frame = camera.read()
            if not success or frame is None:
                print("Failed to read frame")
                time.sleep(0.1)
                continue
                
            # Detect ID cards in the frame
            processed_frame, _ = detect_id_cards_webcam(frame)
            
            # Update the output frame
            with lock:
                output_frame = processed_frame.copy()
                
        print("Webcam stream ended")
    except Exception as e:
        webcam_error = f"Webcam error: {str(e)}"
        print(webcam_error)
        webcam_active = False
    finally:
        # Release resources
        if camera is not None and camera.isOpened():
            camera.release()
            print("Camera released")

def generate():
    global output_frame, lock, webcam_active, webcam_error
    
    while True:
        # If webcam is not active anymore but was supposed to be, return error image
        if not webcam_active and webcam_error:
            error_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_img, webcam_error, (20, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Encode and yield the error image
            (flag, encoded_image) = cv2.imencode(".jpg", error_img)
            if flag:
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                     bytearray(encoded_image) + b'\r\n')
                time.sleep(1.0)  # Slow down error frames
                continue
        
        with lock:
            if output_frame is None:
                continue
                
            # Encode the frame in JPEG format
            (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
                
        # Yield the output frame in byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encoded_image) + b'\r\n')
        time.sleep(0.04)  # ~25 FPS

@app.route('/webcam_feed')
def webcam_feed():
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detection_count')
def get_detection_count():
    global detection_count, webcam_active, webcam_error
    
    if not webcam_active and webcam_error:
        return jsonify({'count': 0, 'error': webcam_error})
        
    return jsonify({'count': detection_count})

@app.route('/webcam_status')
def get_webcam_status():
    global webcam_active, webcam_error
    return jsonify({
        'active': webcam_active,
        'error': webcam_error
    })

@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    global webcam_active, detection_count, webcam_error
    
    if not webcam_active:
        webcam_active = True
        detection_count = 0
        webcam_error = None
        
        # Start webcam in a new thread
        thread = threading.Thread(target=webcam_stream)
        thread.daemon = True
        thread.start()
        
        # Give the webcam a moment to initialize
        time.sleep(0.5)
        
        return jsonify({'success': True, 'message': 'Webcam started'})
    
    return jsonify({'success': False, 'message': 'Webcam already running'})

@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    global webcam_active, detection_count
    
    if webcam_active:
        webcam_active = False
        detection_count = 0
        time.sleep(1)  # Give time for thread to close
        return jsonify({'success': True, 'message': 'Webcam stopped'})
    
    return jsonify({'success': False, 'message': 'Webcam not running'})

@app.route('/test')
def test_page():
    """Route for the image test page"""
    return render_template('image_test.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0') 