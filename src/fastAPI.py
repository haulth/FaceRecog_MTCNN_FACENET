from fastapi import FastAPI, File, UploadFile
from typing import List, Tuple
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import base64

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
