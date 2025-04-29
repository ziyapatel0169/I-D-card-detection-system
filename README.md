# ID Card Detection Web App

This web application allows users to upload images or use a webcam to detect ID cards using a YOLOv8 model.

## Setup Instructions

### Prerequisites
- Python 3.10
- Conda (for virtual environment management)
- Webcam (optional, for live detection)

### Setting up the Environment

1. Clone or download this repository.

2. Create a new Conda environment:
   ```
   conda create -n idcard-detection python=3.10
   ```

3. Activate the environment:
   ```
   conda activate idcard-detection
   ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Running the Application

1. Make sure you're in the project directory and your Conda environment is activated.

2. Run the Flask application:
   ```
   python app.py
   ```

3. Open your web browser and go to http://127.0.0.1:5000/

## Usage

### Image Upload Mode
1. Select "Upload Image" mode (selected by default)
2. Click on the "Choose File" button to select an image containing ID cards
3. Click "Detect ID Cards" to process the image
4. The application will display the original image alongside the processed image with ID cards highlighted
5. The number of detected ID cards will be shown below the images

### Webcam Mode
1. Select "Use Webcam" mode
2. Click "Start Webcam" to access your computer's camera
3. The app will automatically detect ID cards in real-time, with detection results displayed directly on the webcam feed
4. The current count of detected ID cards is shown below the video feed
5. Click "Stop Webcam" when you're done

## Model Information

This application uses a YOLOv8 model trained to detect ID cards. The model was trained on a custom dataset and can identify ID cards in various orientations and lighting conditions.

## Project Structure

- `app.py`: Main Flask application
- `templates/`: HTML templates for the web interface
- `static/`: Static files (CSS, JavaScript)
- `uploads/`: Directory for storing uploaded and processed images
- `runs/train/exp4/weights/best.pt`: Trained YOLOv8 model
- `data/`: Training and validation data 