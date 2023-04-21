from fastapi import FastAPI, File

from starlette.responses import Response
import io,os
from PIL import Image

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from typing import List, Tuple
from io import BytesIO
import base64
from fastapi.responses import StreamingResponse
# Import the face detector and recognizer classes from the provided code
from FacialRecognition import FaceDetector, FaceRecognition, BarcodeReader
import cv2,uvicorn
import numpy as np 
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
    title="Facenet-MTCNN FastAPI",
    description="""Mô hình điểm danh bằng khuôn mặt - deloy with fastapi  """,
    version="1.0.0",
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
def get_image_from_bytes(byte_string):
    img = Image.open(io.BytesIO(byte_string))
    return img


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


@app.post("/object-to-json")
async def detect_food_return_json_result(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    input_image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    faces = detector.get_faces(rgb)
    
    
    predictions = []
    # Detect faces in the frame
    faces, _= detector.get_faces(rgb)
    for face in faces:
        x1, y1, x2, y2 = face[:4]

        # Get face image
        face_img = rgb[int(y1):int(y2), int(x1):int(x2), :]

        # Get face embeddings
        embeddings = detector.get_embeddings(face_img)

        # Recognize face
        name, prob = recognizer.recognize_face(embeddings)
        predictions.append((name, prob))
    # Read barcodes from the image
    barcodes, _ = barcode_reader.read_barcodes(input_image)

    # Convert the image to base64 format
    img_buffer = BytesIO()
    Image.fromarray(input_image).save(img_buffer, format="JPEG")
    img_str = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

    # Return the results
    return {
        "faces": predictions,
        "barcodes": barcodes,
        "image": img_str
    }


@app.post("/object-to-img")
async def detect_food_return_base64_img(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    input_image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    barcodes,img = barcode_reader.read_barcodes(input_image)
    faces = detector.get_faces(img)
    
    predictions = []
    # Detect faces in the frame
    faces, _= detector.get_faces(img)
    for face in faces:
        x1, y1, x2, y2 = face[:4]

        # Get face image
        face_img = img[int(y1):int(y2), int(x1):int(x2), :]

        # Get face embeddings
        embeddings = detector.get_embeddings(face_img)

        # Recognize face
        name, prob = recognizer.recognize_face(embeddings)
        predictions.append((name, prob))
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, "{} {:.2f}".format(name, prob), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # Read barcodes from the image


    # Convert the image to base64 format
    img_buffer = BytesIO()
    Image.fromarray(img).save(img_buffer, format="JPEG")
    img_str = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    #img_base64.save(bytes_io, format="jpeg")
    return Response(content=img_buffer.getvalue(), media_type="image/jpeg")

file_path = '../names.csv'
#hàm lưu tên khi đâu vào là anh người dùng
def write_to_file(name):
    global file_path
    with open(file_path, 'a+') as file:
        file.seek(0)
        names = file.read().splitlines()
        if name not in names and name != 'unknown':
            file.write(name + '\n')
#hàm luu tên khi đâu vào là mã qrcode
def write_barcode_to_file(barcodes):
    global file_path
    with open(file_path, 'a+') as file:
        file.seek(0)
        names = file.read().splitlines()
        if isinstance(barcodes, list):
            barcodes_str = ' '.join(str(code) for code in barcodes)
        else:
            barcodes_str = str(barcodes)
        if barcodes_str not in names and barcodes_str != 'unknown':
            file.write(barcodes_str + '\n')

# uvicorn main:app --reload  --port 8080
@app.get("/camera_feed")
async def camera_feed():
    
    cap = cv2.VideoCapture(0)

    async def generate():
        while True:
            try:
                success, image = cap.read()
                if not success:
                    break

                input_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

                barcodes, img = barcode_reader.read_barcodes(input_image)

                # nhận dạng khuôn mặt
                faces = detector.get_faces(img)

                predictions = []
                # Detect faces in the frame
                faces, _ = detector.get_faces(img)
                for face in faces:
                    x1, y1, x2, y2 = face[:4]

                    # Get face image
                    face_img = img[int(y1):int(y2), int(x1):int(x2), :]

                    # Get face embeddings
                    embeddings = detector.get_embeddings(face_img)

                    # Recognize face
                    name, prob = recognizer.recognize_face(embeddings)
                    predictions.append((name, prob))

                    cv2.rectangle(img, (int(x1), int(y1)),
                                  (int(x2), int(y2)), (0, 255, 0), 2)

                    cv2.putText(img, "{} {:.2f}".format(name, prob), (int(x1), int(
                        y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    write_to_file(name)

                if barcodes:
                    write_barcode_to_file(barcodes)

                frame = cv2.imencode(".jpg", img[:, :, ::-1])[1].tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            except cv2.error as e:
                print(f"Error occurred: {e}")
                cap.release()
                break

    return StreamingResponse(generate(), media_type='multipart/x-mixed-replace; boundary=frame')

