from flask import Flask, jsonify, request
import base64
from FacialRecognition import *

app = Flask(__name__)

detector = FaceDetector(
    minsize=20,
    threshold=[0.6, 0.7, 0.7],
    factor=0.709,
    gpu_memory_fraction=0.6,
    detect_face_model_path="./src/align",
    facenet_model_path="./Models/20180402-114759.pb"
)

recognizer = FaceRecognition(classifier_path="./Models/facemodel.pkl")

barcode_reader = BarcodeReader()

@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    data = request.json
    image = base64.b64decode(data['image'])
    nparr = np.fromstring(image, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    faces = detector.get_faces(frame)
    embeddings = []
    for face in faces:
        (x1, y1, x2, y2, acc) = face
        cropped = frame[int(y1):int(y2), int(x1):int(x2), :]
        embedding = detector.get_embeddings(cropped)
        embeddings.append(embedding)
    if embeddings:
        embeddings = np.concatenate(embeddings)
        name, confidence = recognizer.recognize_face(embeddings)
    else:
        name, confidence = "unknown", 0.0
    return jsonify({"name": name, "confidence": confidence})

@app.route('/api/barcode', methods=['POST'])
def read_barcode():
    data = request.json
    image = base64.b64decode(data['image'])
    nparr = np.fromstring(image, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    barcodes = barcode_reader.read_barcodes(frame)
    return jsonify({"barcodes": barcodes})

if __name__ == '__main__':
    app.run(debug=True)