# Copyright
Nhận diện khuôn mặt khá chuẩn xác bằng MTCNN và Facenet!
Chạy trên Tensorflow 2.x

step 1:
'''
pip install -r requirements.txt
'''

step 2: run
'''
## run with Camera
python face_rec_cam.py 
## run with api flask
pip install flask
python face_rec_flask.py 
## run with api fastapi
start server:
```
uvicorn main:app --reload  --port 8000
```
link access to local server: 
```
http://localhost:8000/docs#/
```
