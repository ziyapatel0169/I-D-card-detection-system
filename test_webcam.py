import cv2
import time

def test_webcam():
    print("Testing webcam access...")
    
    # Try multiple camera indices
    for camera_index in range(3):  # Try indices 0, 1, 2
        print(f"\nTrying camera index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"  Failed to open camera {camera_index}")
            continue
            
        print(f"  Successfully opened camera {camera_index}")
        print("  Camera properties:")
        print(f"  - Width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
        print(f"  - Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        print(f"  - FPS: {cap.get(cv2.CAP_PROP_FPS)}")
        
        # Try to read a frame
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"  Camera {camera_index} opened but couldn't read frame")
            cap.release()
            continue
            
        print(f"  Successfully read frame from camera {camera_index}")
        print(f"  Frame shape: {frame.shape}")
        
        # Save the frame to verify
        output_filename = f"camera_{camera_index}_test.jpg"
        cv2.imwrite(output_filename, frame)
        print(f"  Saved test frame to {output_filename}")
        
        # Release the camera
        cap.release()
        
        print(f"  Camera {camera_index} test complete")

if __name__ == "__main__":
    test_webcam()
    print("\nWebcam test complete. Check the output to see which camera indices work.") 