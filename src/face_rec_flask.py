"""
File: src/face_rec_flask.py
Mô tả: Flask API server cho nhận diện khuôn mặt sử dụng MTCNN + FaceNet + SVM

Chức năng chính:
- Cung cấp REST API để nhận diện khuôn mặt từ ảnh
- Sử dụng MTCNN để phát hiện khuôn mặt
- Sử dụng FaceNet để trích xuất embedding
- Sử dụng SVM classifier để phân loại và nhận diện người

Các API endpoints:
1. GET /test - Health check (kiểm tra server)
2. POST /demo/ - Nhận diện khuôn mặt từ ảnh base64

Cách chạy:
    cd src
    python face_rec_flask.py
    
Server sẽ chạy tại: http://localhost:8000
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask
from flask import render_template , request, Response, send_from_directory, jsonify
from flask_cors import CORS, cross_origin
import tensorflow as tf
import argparse
import facenet
import os
import sys
import math
import pickle
import align.detect_face
import numpy as np
import cv2
import collections
from sklearn.svm import SVC
import base64

# ============================================
# CẤU HÌNH THAM SỐ
# ============================================
MINSIZE = 20  # Kích thước khuôn mặt tối thiểu để phát hiện (pixel)
THRESHOLD = [0.6, 0.7, 0.7]  # Ngưỡng cho 3 mạng của MTCNN (P-Net, R-Net, O-Net)
FACTOR = 0.709  # Hệ số scale cho image pyramid trong MTCNN
IMAGE_SIZE = 182  # Kích thước ảnh sau khi align
INPUT_IMAGE_SIZE = 160  # Kích thước ảnh đầu vào của FaceNet
CLASSIFIER_PATH = 'Models/facemodel.pkl'  # Đường dẫn đến model SVM đã train
FACENET_MODEL_PATH = 'Models/20180402-114759.pb'  # Đường dẫn đến model FaceNet
TRAIN_DATA_DIR = '../DataSet/FaceData/raw'  # Thư mục lưu ảnh training
MIN_IMAGES_PER_PERSON = 5  # Số lượng ảnh tối thiểu mỗi người để train (khuyến nghị: 10-20)

# ============================================
# LOAD SVM CLASSIFIER
# Model này chứa các class names và đã được train sẵn
# ============================================
print("Đang tải SVM classifier...")
with open(CLASSIFIER_PATH, 'rb') as file:
    model, class_names = pickle.load(file)
print("Custom Classifier, Successfully loaded")

# ============================================
# KHỞI TẠO TENSORFLOW SESSION
# ============================================
tf.compat.v1.Graph().as_default()

# Cấu hình GPU (sử dụng 60% bộ nhớ GPU nếu có)
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

# ============================================
# LOAD FACENET MODEL
# Model này dùng để trích xuất embedding (vector đặc trưng) từ khuôn mặt
# ============================================
print('Đang tải FaceNet model...')
facenet.load_model(FACENET_MODEL_PATH)

# Lấy các tensor input/output từ graph đã load
tf.compat.v1.disable_eager_execution()
images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]

# ============================================
# LOAD MTCNN MODEL (Face Detection)
# MTCNN gồm 3 mạng: P-Net, R-Net, O-Net để phát hiện khuôn mặt
# ============================================
print('Đang tải MTCNN model...')
pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "./align")
print('Tất cả models đã được load thành công!')



app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)  # Cho phép CORS để frontend có thể gọi API từ domain khác

# Biến global để lưu camera
camera = None

# ============================================
# HÀM HỖ TRỢ
# ============================================
def reload_classifier():
    """Reload lại classifier sau khi train"""
    global model, class_names
    with open(CLASSIFIER_PATH, 'rb') as file:
        model, class_names = pickle.load(file)
    print("Classifier đã được reload!")

def train_classifier():
    """Train lại classifier từ dữ liệu trong thư mục training"""
    try:
        import facenet
        from facenet import ImageClass
        
        # Lấy danh sách thư mục (mỗi thư mục là một người)
        dataset = facenet.get_dataset(TRAIN_DATA_DIR)
        
        if len(dataset) == 0:
            return {"success": False, "message": "Không tìm thấy dữ liệu training"}
        
        # Lấy paths và labels
        paths, labels = facenet.get_image_paths_and_labels(dataset)
        
        if len(paths) == 0:
            return {"success": False, "message": "Không có ảnh để train"}
        
        # Kiểm tra số lượng ảnh tối thiểu cho mỗi người
        persons_with_few_images = []
        for cls in dataset:
            if len(cls.image_paths) < MIN_IMAGES_PER_PERSON:
                persons_with_few_images.append(f"{cls.name} ({len(cls.image_paths)} ảnh)")
        
        if persons_with_few_images:
            warning_msg = f"Cảnh báo: {len(persons_with_few_images)} người có ít hơn {MIN_IMAGES_PER_PERSON} ảnh:\n"
            warning_msg += ", ".join(persons_with_few_images[:5])  # Hiển thị tối đa 5 người
            if len(persons_with_few_images) > 5:
                warning_msg += f" và {len(persons_with_few_images) - 5} người khác"
            warning_msg += f"\n\nKhuyến nghị: Mỗi người cần ít nhất {MIN_IMAGES_PER_PERSON} ảnh để nhận diện chính xác."
            warning_msg += "\nTối thiểu: 5-10 ảnh/người | Khuyến nghị: 20-30 ảnh/người | Tốt nhất: 50+ ảnh/người"
            print(f"WARNING: {warning_msg}")
        
        print(f'Đang train với {len(dataset)} người, {len(paths)} ảnh')
        
        # Tính embeddings cho tất cả ảnh
        nrof_images = len(paths)
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / 90))
        emb_array = np.zeros((nrof_images, embedding_size))
        
        for i in range(nrof_batches_per_epoch):
            start_index = i * 90
            end_index = min((i + 1) * 90, nrof_images)
            paths_batch = paths[start_index:end_index]
            images = facenet.load_data(paths_batch, False, False, 160)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
        
        # Train SVM
        print('Đang train SVM classifier...')
        new_model = SVC(kernel='linear', probability=True)
        new_model.fit(emb_array, labels)
        
        # Tạo class names
        class_names = [cls.name.replace('_', ' ') for cls in dataset]
        
        # Lưu model
        with open(CLASSIFIER_PATH, 'wb') as outfile:
            pickle.dump((new_model, class_names), outfile)
        
        # Reload model
        reload_classifier()
        
        train_accuracy = np.mean(new_model.predict(emb_array) == labels)
        
        # Tính số lượng ảnh trung bình mỗi người
        avg_images_per_person = len(paths) / len(dataset) if len(dataset) > 0 else 0
        
        # Tạo thông báo chi tiết
        message = f"Train thành công! Accuracy: {train_accuracy:.3f}\n"
        message += f"Số người: {len(dataset)} | Tổng ảnh: {len(paths)} | Trung bình: {avg_images_per_person:.1f} ảnh/người"
        
        if avg_images_per_person < MIN_IMAGES_PER_PERSON:
            message += f"\n Cảnh báo: Số ảnh trung bình ({avg_images_per_person:.1f}) thấp hơn khuyến nghị ({MIN_IMAGES_PER_PERSON}+)."
            message += "\nĐể cải thiện độ chính xác, hãy upload thêm ảnh cho mỗi người (nhiều góc độ, ánh sáng khác nhau)."
        
        if persons_with_few_images:
            message += f"\n\n {len(persons_with_few_images)} người có ít hơn {MIN_IMAGES_PER_PERSON} ảnh - độ chính xác có thể thấp."
        
        return {
            "success": True, 
            "message": message,
            "nrof_classes": len(dataset),
            "nrof_images": len(paths),
            "avg_images_per_person": round(avg_images_per_person, 1),
            "train_accuracy": round(train_accuracy, 3),
            "warning": len(persons_with_few_images) > 0
        }
    except Exception as e:
        return {"success": False, "message": f"Lỗi khi train: {str(e)}"}

def get_camera():
    """Lấy camera instance (singleton pattern)"""
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return camera

# ============================================
# API 1: GET / - Serve HTML file
# Chức năng: Trả về giao diện web
# ============================================
@app.route('/')
@cross_origin()
def home():
    return send_from_directory('static', 'index.html')

# ============================================
# API 2: GET /test
# Chức năng: Kiểm tra server có hoạt động không (Health check)
# Input: Không cần
# Output: "OK!" nếu server đang chạy
# ============================================
@app.route('/test')
@cross_origin()
def index():
    return "OK!"


# ============================================
# API 2: POST /demo/
# Chức năng: Nhận diện khuôn mặt từ ảnh được gửi dưới dạng base64
# Input (form-data):
#   - image: Chuỗi base64 của ảnh
#   - w: Chiều rộng ảnh (width)
#   - h: Chiều cao ảnh (height)
# Output: Tên người được nhận diện hoặc "Unknown" nếu không nhận diện được
# 
# Quy trình xử lý:
#   1. Nhận ảnh base64 từ client
#   2. Giải mã base64 thành ảnh numpy array
#   3. Sử dụng MTCNN để phát hiện khuôn mặt trong ảnh
#   4. Với mỗi khuôn mặt tìm được:
#      - Resize về kích thước 160x160
#      - Prewhiten (chuẩn hóa)
#      - Đưa vào FaceNet để trích xuất embedding (vector đặc trưng)
#      - Sử dụng SVM classifier để phân loại và dự đoán tên người
#   5. Trả về tên người nếu xác suất > 0.5, ngược lại trả "Unknown"
# ============================================
@app.route('/demo/', methods=['POST'])
@cross_origin()
def upload_img_file():
    if request.method == 'POST':
        # Khởi tạo tên mặc định là "Unknown"
        name="Unknown"
        
        # Lấy dữ liệu từ form: ảnh base64, chiều rộng và chiều cao
        f = request.form.get('image')  # Chuỗi base64 của ảnh
        w = int(request.form.get('w'))  # Chiều rộng ảnh
        h = int(request.form.get('h'))  # Chiều cao ảnh

        # Giải mã chuỗi base64 thành dữ liệu ảnh nhị phân
        decoded_string = base64.b64decode(f)
        # Chuyển đổi dữ liệu nhị phân thành numpy array
        frame = np.fromstring(decoded_string, dtype=np.uint8)
        # Giải mã ảnh từ buffer (hỗ trợ nhiều định dạng: JPEG, PNG, etc.)
        frame = cv2.imdecode(frame, cv2.IMREAD_ANYCOLOR)

        # Sử dụng MTCNN để phát hiện khuôn mặt trong ảnh
        # Trả về bounding boxes (tọa độ các khuôn mặt) và landmarks
        bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

        # Đếm số lượng khuôn mặt phát hiện được
        faces_found = bounding_boxes.shape[0]

        # Nếu tìm thấy ít nhất 1 khuôn mặt
        if faces_found > 0:
            # Lấy tọa độ bounding box của khuôn mặt đầu tiên
            det = bounding_boxes[:, 0:4]
            bb = np.zeros((faces_found, 4), dtype=np.int32)
            for i in range(faces_found):
                bb[i][0] = det[i][0]  # x1
                bb[i][1] = det[i][1]  # y1
                bb[i][2] = det[i][2]  # x2
                bb[i][3] = det[i][3]  # y2
                
                # Lấy vùng ảnh chứa khuôn mặt (hiện tại đang dùng toàn bộ ảnh)
                cropped = frame
              
                #cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                
                # Resize ảnh về kích thước 160x160 (kích thước đầu vào của FaceNet)
                scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                    interpolation=cv2.INTER_CUBIC)
                # Chuẩn hóa ảnh (prewhiten) để cải thiện độ chính xác
                scaled = facenet.prewhiten(scaled)
                # Reshape để phù hợp với input của FaceNet: (batch_size, 160, 160, 3)
                scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                
                # Đưa ảnh vào FaceNet để trích xuất embedding (vector đặc trưng 128 chiều)
                feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                emb_array = sess.run(embeddings, feed_dict=feed_dict)
                
                # Sử dụng SVM classifier để dự đoán tên người từ embedding
                predictions = model.predict_proba(emb_array)
                # Tìm class có xác suất cao nhất
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[
                    np.arange(len(best_class_indices)), best_class_indices]
                best_name = class_names[best_class_indices[0]]
                print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))

                # Chỉ trả về tên nếu xác suất > 0.5 (50%), ngược lại trả "Unknown"
                if best_class_probabilities > 0.5:
                    name = class_names[best_class_indices[0]]
                else:
                    name = "Unknown"

        # Trả về tên người được nhận diện
        return name


# ============================================
# API 3: POST /train
# Chức năng: Upload ảnh và train model với tên người
# Input (form-data):
#   - image: File ảnh
#   - name: Tên người (tên thư mục sẽ được tạo)
# Output: JSON với thông tin kết quả train
# ============================================
@app.route('/train', methods=['POST'])
@cross_origin()
def train_face():
    try:
        print("\n" + "="*60)
        print("[UPLOAD] BAT DAU XU LY UPLOAD ANH")
        print("="*60)
        
        if 'image' not in request.files or 'name' not in request.form:
            print("[ERROR] Thieu anh hoac ten nguoi")
            return jsonify({"success": False, "message": "Thiếu ảnh hoặc tên người"})
        
        file = request.files['image']
        person_name = request.form['name'].strip()
        
        print(f"[INFO] Ten nguoi: {person_name}")
        print(f"[FILE] Ten file: {file.filename}")
        
        if file.filename == '' or person_name == '':
            print("[ERROR] Ten nguoi khong duoc de trong")
            return jsonify({"success": False, "message": "Tên người không được để trống"})
        
        # Tạo thư mục cho người này nếu chưa có
        # Sử dụng đường dẫn tuyệt đối để tránh lỗi
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        train_data_dir = os.path.join(base_dir, 'DataSet', 'FaceData', 'raw')
        person_dir = os.path.join(train_data_dir, person_name)
        
        print(f"[DIR] Thu muc base: {base_dir}")
        print(f"[DIR] Thu muc train data: {train_data_dir}")
        print(f"[DIR] Thu muc nguoi: {person_dir}")
        
        # Tạo thư mục với encoding UTF-8
        os.makedirs(person_dir, exist_ok=True)
        print(f"[OK] Da tao/kiem tra thu muc: {person_dir}")
        
        # Đọc ảnh
        file_bytes = file.read()
        print(f"[DATA] Kich thuoc file: {len(file_bytes)} bytes")
        
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            print("[ERROR] Khong the doc anh")
            return jsonify({"success": False, "message": "Không thể đọc ảnh"})
        
        print(f"[OK] Da doc anh, kich thuoc: {img.shape}")
        
        # Phát hiện khuôn mặt
        print("[DETECT] Dang phat hien khuon mat...")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bounding_boxes, _ = align.detect_face.detect_face(rgb, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
        
        print(f"[DETECT] So khuon mat phat hien: {bounding_boxes.shape[0]}")
        
        if bounding_boxes.shape[0] == 0:
            print("[ERROR] Khong phat hien duoc khuon mat trong anh")
            return jsonify({"success": False, "message": "Không phát hiện được khuôn mặt trong ảnh"})
        
        # Lấy khuôn mặt đầu tiên
        det = bounding_boxes[0, 0:4]
        x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
        print(f"[COORD] Toa do khuon mat: ({x1}, {y1}, {x2}, {y2})")
        
        # Cắt và resize khuôn mặt
        face_img = rgb[y1:y2, x1:x2, :]
        face_resized = cv2.resize(face_img, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
        print(f"[OK] Da resize khuon mat: {face_resized.shape}")
        
        # Lưu ảnh với encoding UTF-8
        face_bgr = cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR)
        
        # Đếm số ảnh hiện có
        existing_images = len([f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        print(f"[DATA] So anh hien co: {existing_images}")
        
        img_filename = f"{person_name}_{existing_images + 1}.jpg"
        img_path = os.path.join(person_dir, img_filename)
        
        print(f"[SAVE] Duong dan luu: {img_path}")
        
        # Sử dụng cv2.imencode để tránh lỗi encoding với tên file tiếng Việt
        # Sau đó lưu bằng numpy
        success, encoded_image = cv2.imencode('.jpg', face_bgr)
        if not success:
            print("[ERROR] Khong the encode anh")
            return jsonify({"success": False, "message": "Không thể encode ảnh"})
        
        # Lưu file bằng cách ghi trực tiếp byte array
        with open(img_path, 'wb') as f:
            f.write(encoded_image.tobytes())
        
        print(f"[OK] Da luu anh thanh cong!")
        
        # Kiểm tra file đã được lưu chưa
        if os.path.exists(img_path):
            file_size = os.path.getsize(img_path)
            print(f"[OK] Xac nhan: File ton tai, kich thuoc: {file_size} bytes")
        else:
            print(f"[WARNING] File khong ton tai sau khi luu!")
        
        # Liệt kê tất cả file trong thư mục
        files_in_dir = os.listdir(person_dir)
        print(f"[FILES] Cac file trong thu muc: {files_in_dir}")
        
        print("="*60)
        print("[UPLOAD] KET THUC XU LY UPLOAD ANH")
        print("="*60 + "\n")
        
        # Chỉ lưu ảnh, không train ngay (sẽ train sau khi upload hết)
        return jsonify({
            "success": True,
            "message": f"Đã lưu ảnh thành công! ({existing_images + 1} ảnh)",
            "person_name": person_name,
            "image_path": img_path,
            "total_images": existing_images + 1
        })
            
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[EXCEPTION] {str(e)}")
        print(f"[TRACE] Chi tiet loi:\n{error_trace}")
        return jsonify({"success": False, "message": f"Lỗi: {str(e)}", "traceback": error_trace})

# ============================================
# API 4: POST /retrain
# Chức năng: Train lại model từ tất cả ảnh trong dataset
# Output: JSON với thông tin kết quả train
# ============================================
@app.route('/retrain', methods=['POST'])
@cross_origin()
def retrain_model():
    """Train lại model từ toàn bộ dataset"""
    result = train_classifier()
    return jsonify(result)


# ============================================
# API 5: GET /train_visualization
# Chức năng: Lấy dữ liệu để visualize SVM model
# Output: JSON với embeddings 2D (PCA), labels, class names
# ============================================
@app.route('/train_visualization', methods=['GET'])
@cross_origin()
def get_train_visualization():
    """Trả về dữ liệu visualization của SVM model"""
    try:
        from sklearn.decomposition import PCA
        import facenet
        from facenet import ImageClass
        
        print("\n[VIZ] Bat dau tao du lieu visualization...")
        
        # Lấy dataset
        dataset = facenet.get_dataset(TRAIN_DATA_DIR)
        
        if len(dataset) == 0:
            return jsonify({"success": False, "message": "Không có dữ liệu training"})
        
        # Lấy paths và labels
        paths, labels = facenet.get_image_paths_and_labels(dataset)
        
        if len(paths) == 0:
            return jsonify({"success": False, "message": "Không có ảnh trong dataset"})
        
        print(f"[VIZ] So nguoi: {len(dataset)}, So anh: {len(paths)}")
        
        # Tính embeddings cho tất cả ảnh
        nrof_images = len(paths)
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / 90))
        emb_array = np.zeros((nrof_images, embedding_size))
        
        for i in range(nrof_batches_per_epoch):
            start_index = i * 90
            end_index = min((i + 1) * 90, nrof_images)
            paths_batch = paths[start_index:end_index]
            images = facenet.load_data(paths_batch, False, False, 160)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
        
        print(f"[VIZ] Da tinh {nrof_images} embeddings (128D)")
        
        # Giảm chiều từ 128D xuống 2D bằng PCA để visualize
        if nrof_images >= 2:
            pca = PCA(n_components=2)
            emb_2d = pca.fit_transform(emb_array)
            print(f"[VIZ] Da giam chieu xuong 2D bang PCA")
            print(f"[VIZ] Explained variance ratio: {pca.explained_variance_ratio_}")
        else:
            return jsonify({"success": False, "message": "Cần ít nhất 2 ảnh để visualize"})
        
        # Tạo class names
        class_names = [cls.name.replace('_', ' ') for cls in dataset]
        
        # Tạo màu sắc cho từng class
        colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
            '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52B788',
            '#E63946', '#F77F00', '#06FFC5', '#118AB2', '#073B4C'
        ]
        
        # Chuẩn bị dữ liệu trả về
        data_points = []
        for i in range(nrof_images):
            label = labels[i]
            data_points.append({
                'x': float(emb_2d[i, 0]),
                'y': float(emb_2d[i, 1]),
                'label': label,
                'class_name': class_names[label],
                'color': colors[label % len(colors)]
            })
        
        # Tính decision boundary bằng cách train SVM trên không gian 2D
        decision_boundary = None
        if len(dataset) >= 2 and nrof_images >= 10:
            try:
                # Train một SVM mới trên không gian 2D để visualize decision boundary
                print("[VIZ] Dang train SVM tren khong gian 2D de ve decision boundary...")
                svm_2d = SVC(kernel='linear', probability=True)
                svm_2d.fit(emb_2d, labels)
                
                # Tạo grid để vẽ decision boundary
                x_min, x_max = emb_2d[:, 0].min() - 1, emb_2d[:, 0].max() + 1
                y_min, y_max = emb_2d[:, 1].min() - 1, emb_2d[:, 1].max() + 1
                h = (x_max - x_min) / 100  # resolution
                
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                     np.arange(y_min, y_max, h))
                
                # Predict trên grid
                Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                
                # Chuyển thành list để trả về JSON
                decision_boundary = {
                    'x_min': float(x_min),
                    'x_max': float(x_max),
                    'y_min': float(y_min),
                    'y_max': float(y_max),
                    'resolution': float(h),
                    'predictions': Z.tolist(),
                    'shape': list(Z.shape)
                }
                
                print(f"[VIZ] Da tinh decision boundary voi grid size: {Z.shape}")
                
                # Lấy coefficients (w) và intercept (b) để vẽ đường thẳng
                # Với SVM linear: w[0]*x + w[1]*y + b = 0
                hyperplanes = []
                
                if len(dataset) == 2:
                    # Binary classification - 1 hyperplane
                    w = svm_2d.coef_[0]
                    b = svm_2d.intercept_[0]
                    hyperplanes.append({
                        'w': [float(w[0]), float(w[1])],
                        'b': float(b),
                        'class1': 0,
                        'class2': 1
                    })
                    print(f"[VIZ] Hyperplane: w={w}, b={b}")
                else:
                    # Multi-class - nhiều hyperplanes (one-vs-rest)
                    # Mỗi hyperplane tách 1 class vs tất cả class còn lại
                    class_names_list = [cls.name for cls in dataset]
                    for i in range(len(svm_2d.coef_)):
                        w = svm_2d.coef_[i]
                        b = svm_2d.intercept_[i]
                        hyperplanes.append({
                            'w': [float(w[0]), float(w[1])],
                            'b': float(b),
                            'class_id': i,
                            'class_name': class_names_list[i] if i < len(class_names_list) else f'class_{i}'
                        })
                        print(f"[VIZ] Hyperplane {i} ({class_names_list[i] if i < len(class_names_list) else i}): w=[{w[0]:.4f}, {w[1]:.4f}], b={b:.4f}")
                    print(f"[VIZ] So hyperplanes: {len(hyperplanes)}")
                
                decision_boundary['hyperplanes'] = hyperplanes
                
                # Tính support vectors nếu có
                if hasattr(svm_2d, 'support_vectors_'):
                    support_vectors = svm_2d.support_vectors_.tolist()
                    decision_boundary['support_vectors'] = support_vectors
                    print(f"[VIZ] So support vectors: {len(support_vectors)}")
                    
            except Exception as e:
                print(f"[VIZ] Loi khi tinh decision boundary: {e}")
                decision_boundary = None
        
        # Thống kê theo class
        class_stats = []
        for idx, cls in enumerate(dataset):
            class_data = [p for p in data_points if p['label'] == idx]
            class_stats.append({
                'name': cls.name.replace('_', ' '),
                'count': len(cls.image_paths),
                'color': colors[idx % len(colors)]
            })
        
        print("[VIZ] Hoan thanh tao du lieu visualization")
        
        return jsonify({
            'success': True,
            'data_points': data_points,
            'class_stats': class_stats,
            'total_classes': len(dataset),
            'total_images': nrof_images,
            'explained_variance': [float(v) for v in pca.explained_variance_ratio_],
            'model_type': 'SVM Linear Kernel',
            'decision_boundary': decision_boundary
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[VIZ ERROR] {str(e)}")
        print(f"[TRACE] {error_trace}")
        return jsonify({"success": False, "message": f"Lỗi: {str(e)}"})


# ============================================
# API 5: GET /camera_feed
# Chức năng: Stream camera real-time với nhận diện khuôn mặt
# Output: Video stream (MJPEG)
# ============================================
def generate_frames():
    """Generator function để stream video frames"""
    cap = get_camera()
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        try:
            # Chuyển BGR sang RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Phát hiện khuôn mặt
            bounding_boxes, _ = align.detect_face.detect_face(rgb, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
            faces_found = bounding_boxes.shape[0]
            
            # Nhận diện từng khuôn mặt
            if faces_found > 0:
                for i in range(faces_found):
                    det = bounding_boxes[i, 0:4]
                    x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                    
                    # Lấy ảnh khuôn mặt
                    face_img = rgb[y1:y2, x1:x2, :]
                    
                    # Resize và prewhiten
                    scaled = cv2.resize(face_img, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
                    scaled = facenet.prewhiten(scaled)
                    scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                    
                    # Trích xuất embedding
                    feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                    emb_array = sess.run(embeddings, feed_dict=feed_dict)
                    
                    # Dự đoán
                    predictions = model.predict_proba(emb_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                    
                    # Vẽ bounding box và tên
                    if best_class_probabilities[0] > 0.5:
                        name = class_names[best_class_indices[0]]
                        color = (0, 255, 0)  # Xanh lá
                    else:
                        name = "Unknown"
                        color = (0, 0, 255)  # Đỏ
                    
                    # Vẽ lên frame (chuyển về BGR để vẽ)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{name} {best_class_probabilities[0]:.2f}", 
                              (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Encode frame thành JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            
            # Yield frame theo định dạng MJPEG
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        except Exception as e:
            print(f"Lỗi khi xử lý frame: {e}")
            continue

@app.route('/camera_feed')
@cross_origin()
def camera_feed():
    """Stream camera với nhận diện real-time"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ============================================
# CHẠY FLASK SERVER
# ============================================
if __name__ == '__main__':
    # Chạy server Flask
    # debug=True: Tự động reload khi code thay đổi
    # host='0.0.0.0': Cho phép truy cập từ mọi địa chỉ IP
    # port='8000': Chạy trên cổng 8000
    print("=" * 50)
    print("Flask API Server đang khởi động...")
    print("Giao diện web: http://localhost:8000/")
    print("API test: GET http://localhost:8000/test")
    print("API nhận diện: POST http://localhost:8000/demo/")
    print("API train: POST http://localhost:8000/train")
    print("API camera: GET http://localhost:8000/camera_feed")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0',port='8000')

