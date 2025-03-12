from flask import Flask, render_template, Response, request, redirect, url_for, jsonify, send_from_directory
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Specify paths to the YOLOv3 model files
config_file = r'S:\hack\yolov3.cfg'
weights_file = r'S:\hack\yolov3.weights'
labels_file = r'S:\hack\coco.names'

# Load the pre-trained YOLO model from the configuration and weights files
net = cv2.dnn.readNet(weights_file, config_file)

# Load COCO dataset labels
with open(labels_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Global variable to store current predictions
current_predictions = []

# Function to perform object detection
def detect_objects(frame):
    global current_predictions
    height, width = frame.shape[:2]

    # Prepare the frame for YOLO input
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get the output layer names for YOLOv3
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Perform forward pass
    detections = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    current_predictions = []  # Reset predictions for each frame

    # Parse detections and extract relevant info
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                box = obj[0:4] * np.array([width, height, width, height])
                center_x, center_y, box_width, box_height = box.astype('int')

                x = int(center_x - box_width / 2)
                y = int(center_y - box_height / 2)

                boxes.append([x, y, int(box_width), int(box_height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-Max Suppression to eliminate redundant boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Extract remaining boxes and labels
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i] * 100, 2))
            prediction = f'{label} {confidence}%'
            current_predictions.append(prediction)

            # Draw bounding box and label on frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, prediction, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame

# Function to generate frames for video feed
def gen_frames():
    video = cv2.VideoCapture(0)
    try:
        while True:
            success, frame = video.read()
            if not success:
                break
            else:
                frame = detect_objects(frame)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        video.release()

# Route for live video feed (Webcam)
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to serve predictions
@app.route('/predictions')
def get_predictions():
    return jsonify(predictions=current_predictions)

# Utility function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Image upload and detection
@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Load the image and detect objects
            image = cv2.imread(filepath)
            image = detect_objects(image)

            # Save the processed image
            processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + filename)
            cv2.imwrite(processed_filepath, image)

            return redirect(url_for('display_image', filename='processed_' + filename))

    return render_template('upload.html')

# Display the processed image
@app.route('/uploads/<filename>')
def display_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Main route to choose between webcam and image upload
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Create the upload folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
