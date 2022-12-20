from fastapi import FastAPI, File
from segmentation import get_result_cls, get_image_from_bytes
from starlette.responses import Response
import io
from PIL import Image
import json
from fastapi.middleware.cors import CORSMiddleware
import uvicorn 



app = FastAPI(
    title="MTCNN with FaceNet",
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


@app.post("/object-to-json")
async def detect_food_return_json_result(file: bytes = File(...)):
    detect_res=get_result_cls(file)
    return {"result": detect_res}


#@app.post("/object-to-img")
# async def detect_food_return_base64_img(file: bytes = File(...)):
#     input_image = get_image_from_bytes(file)
#     results = model(input_image)
#     results.render()  # updates results.imgs with boxes and labels
#     for img in results.ims:
#         bytes_io = io.BytesIO()
#         img_base64 = Image.fromarray(img)
#         img_base64.save(bytes_io, format="jpeg")
#     return Response(content=bytes_io.getvalue(), media_type="image/jpeg")
