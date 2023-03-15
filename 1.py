from fastapi import FastAPI, File, UploadFile
from typing import List, Tuple
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import base64
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
# Import the face detector and recognizer classes from the provided code
from FacialRecognition import FaceDetector, FaceRecognition, BarcodeReader

app = FastAPI()

# Load face detector and recognizer
detector = FaceDetector(
    minsize=20,
    threshold=[0.6, 0.7, 0.7],
    factor=0.709,
    gpu_memory_fraction=0.6,
    detect_face_model_path="./src/align",
    facenet_model_path="./Models/20180402-114759.pb"
)
recognizer = FaceRecognition(classifier_path="./Models/facemodel.pkl")
# Initialize barcode reader
barcode_reader = BarcodeReader(verbose=False)

app = FastAPI(
    title="Custom YOLOV5 API",
    description="""Mô hình điểm danh bằng khuôn mặt - deloy with fastapi  """,
    version="1.0.1",
)

origins = [
    "http://localhost",
    "http://localhost:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/notify/v1/health')
def get_health():
    """
    Usage on K8S
    readinessProbe:
        httpGet:
            path: /notify/v1/health
            port: 80
    livenessProbe:
        httpGet:
            path: /notify/v1/health
            port: 80
    :return:
        dict(msg='OK')
    """
    return dict(msg='OK')

@app.post("/recognize_faces")
async def recognize_faces(file: UploadFile = File(...)) -> List[Tuple[str, float]]:
    # Read the image from the upload file
    contents = await file.read()
    image = np.array(Image.open(BytesIO(contents)))

    # Get faces from the image
    faces = detector.get_faces(image)

    # Get embeddings for each face
    embeddings = []
    for face in faces:
        # Convert the face to RGB format
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        # Get the embedding for the face
        embedding = detector.get_embeddings(face)
        embeddings.append(embedding)

    # Recognize each face
    predictions = []
    for embedding in embeddings:
        # Recognize the face
        name, confidence = recognizer.recognize_face(embedding)
        predictions.append((name, confidence))

    return predictions

@app.post("/read_barcodes")
async def read_barcodes(file: UploadFile = File(...)) -> List[str]:
    # Read the image from the upload file
    contents = await file.read()
    image = np.array(Image.open(BytesIO(contents)))

    # Read barcodes from the image
    barcodes = barcode_reader.read_barcodes(image)

    return barcodes

@app.post("/process_image")
async def process_image(file: UploadFile = File(...)) -> dict:
    # Read the image from the upload file
    contents = await file.read()
    image = np.array(Image.open(BytesIO(contents)))

    # Get faces from the image
    faces = detector.get_faces(image)

    # Get embeddings for each face
    embeddings = []
    for face in faces:
        # Convert the face to RGB format
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        # Get the embedding for the face
        embedding = detector.get_embeddings(face)
        embeddings.append(embedding)

    # Recognize each face
    predictions = []
    for embedding in embeddings:
        # Recognize the face
        name, confidence = recognizer.recognize_face(embedding)
        predictions.append((name, confidence))

    # Read barcodes from the image
    barcodes = barcode_reader.read_barcodes(image)

    # Convert the image to base64 format
    img_buffer = BytesIO()
    Image.fromarray(image).save(img_buffer, format="JPEG")
    img_str = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

    # Return the results
    return {
        "faces": predictions,
        "barcodes": barcodes,
        "image": img_str
    }

@app.get("/video_feed")
async def video_feed():
    #yêu cầu quyền truy cập camera
    import os

    # Initialize the video capture object
    cap = cv2.VideoCapture(0)

    # Get camera parameters
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize the output buffer
    ret, buffer = cap.read()

    while ret:
        # Convert the buffer to RGB format
        frame = cv2.cvtColor(buffer, cv2.COLOR_BGR2RGB)

        # Get faces from the frame
        faces = detector.get_faces(frame)

        # Get embeddings for each face
        embeddings = []
        for face in faces:
            # Convert the face to RGB format
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            # Get the embedding for the face
            embedding = detector.get_embeddings(face)
            embeddings.append(embedding)

        # Recognize each face
        for i, embedding in enumerate(embeddings):
            # Recognize the face
            name, confidence = recognizer.recognize_face(embedding)

            # Draw a rectangle around the face
            (x, y, w, h) = faces[i]
            cv2.rectangle(buffer, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Add the name and confidence level to the rectangle
            text = f"{name} ({confidence:.2f})"
            cv2.putText(buffer, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Convert the buffer back to BGR format for display
        ret, buffer = cv2.imencode(".jpg", cv2.cvtColor(buffer, cv2.COLOR_RGB2BGR))
        frame = buffer.tobytes()

        # Yield the frame for display
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

        # Read the next frame from the camera
        ret, buffer = cap.read()

    # Release the camera and resources
    cap.release()
# Define a generator function to capture frames from the webcam
async def webcam_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame
    cap.release()

# Define an async function to perform face recognition on a single frame
async def recognize_faces_from_frame(frame):
    # Get faces from the image
    faces = detector.get_faces(frame)

    # Get embeddings for each face
    embeddings = []
    for face in faces:
        # Convert the face to RGB format
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        # Get the embedding for the face
        embedding = detector.get_embeddings(face)
        embeddings.append(embedding)

    # Recognize each face
    predictions = []
    for embedding in embeddings:
        # Recognize the face
        name, confidence = recognizer.recognize_face(embedding)
        predictions.append((name, confidence))

    return predictions

# Define an async function to read barcodes from a single frame
async def read_barcodes_from_frame(frame):
    # Read barcodes from the image
    barcodes = barcode_reader.read_barcodes(frame)

    return barcodes

# Define an endpoint to stream frames from the webcam and perform face recognition on them
@app.get("/webcam_stream")
async def webcam_stream():
    return StreamingResponse(webcam_frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

# Define an async generator function to capture frames from the webcam and perform face recognition on them
async def webcam_frame_generator():
    async for frame in webcam_frames():
        # Perform face recognition and barcode reading on the frame
        face_predictions = await recognize_faces_from_frame(frame)
        barcode_predictions = await read_barcodes_from_frame(frame)

        # Draw the predictions on the frame
        for name, confidence in face_predictions:
            if name:
                cv2.putText(frame, name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, str(confidence), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        for barcode in barcode_predictions:
            cv2.putText(frame, barcode, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
