from typing import Union
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import time


# MediaPipe 모델 설정
base_options = python.BaseOptions(model_asset_path='models/efficientnet_lite0.tflite')
options = vision.ImageClassifierOptions(base_options=base_options, max_results=3)
classifier = vision.ImageClassifier.create_from_options(options)

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int = 1, q: Union[str, None] = "test"):
    return {"item_id": item_id, "q": q}

@app.post("/img_cls")
async def img_cls(
    image: UploadFile = File(...)
):
    # 이미지 파일 저장
    contents = await image.read()
    filename = f"temp_{image.filename}"
    with open(filename, "wb") as f:
        f.write(contents)
    
    
    # 이미지 로드 및 분류
    mp_image = mp.Image.create_from_file(filename)
    classification_result = classifier.classify(mp_image)

    # 결과 추출
    top_category = classification_result.classifications[0].categories[0]
    print(f"{top_category.category_name} ({top_category.score:.2f})")

    # 임시 파일 삭제
    os.remove(filename)
    
    return JSONResponse(
        content={
            "message": "이미지 분류 요청이 성공적으로 처리되었습니다",
            "image_filename": image.filename,
            "top_category": top_category.category_name,
            "score": float(top_category.score)  # float32를 JSON 직렬화 가능한 형태로 변환
        },
        status_code=200
    )


@app.post("/face_recognition")
async def face_recognition(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...)
):
    """
    얼굴 인식 API
    
    두 개의 이미지를 업로드하여 얼굴 인식을 수행합니다.
    
    - **image1**: 첫 번째 이미지 파일
    - **image2**: 두 번째 이미지 파일
    
    Returns:
        JSON 응답
    """
    # 여기에 얼굴 인식 로직을 구현할 수 있습니다
    app.prepare(ctx_id=0, det_size=(640,640))

    # # STEP 3: load data
    # img1 = cv2.imread('iu1.jpg')
    # img2 = cv2.imread('20160310000359_0.jpg')
    # img = ins_get_image('t1')

    # STEP 4: inference
    faces1 = app.get(img1)
    assert len(faces1)==1

    faces2 = app.get(img2)
    assert len(faces2)==1
    pass
    
    return JSONResponse(
        content={
            "message": "얼굴 인식 요청이 성공적으로 처리되었습니다",
            "image1_filename": image1.filename,
            "image2_filename": image2.filename,
        },
        status_code=200
    )