from typing import Union
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2 as cv2

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse

import argparse
import cv2
import sys
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

from transformers import pipeline

review_classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def visualize(
    image,
    detection_result
) -> np.ndarray:
  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)
    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
  return image


base_options = python.BaseOptions(model_asset_path='models\efficientdet_lite0.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.3)
detector = vision.ObjectDetector.create_from_options(options)

# 얼굴 인식 전역 선언
fa_model = FaceAnalysis()
fa_model.prepare(ctx_id=0, det_size=(640,640))


app = FastAPI()

@app.get("/review_classification")
def review_classification(input_text):
    result = review_classifier(input_text)
    return {"classification": result}

@app.post("/img_det")
async def img_det(
    image: UploadFile = File(...)
):
    contents = await image.read()
    filename = f"temp_{image.filename}"
    with open(filename, "wb") as f:
        f.write(contents)
    
    # STEP 3: Load the input image.
    image = mp.Image.create_from_file(filename)
    # STEP 4: Detect objects in the input image.
    detection_result = detector.detect(image)
    # print(detection_result)
    # STEP 5: Process the detection result. In this case, visualize it.
    objects = []
    for detection in detection_result.detections:
        objects.append(detection)
    print(f"Find Object : {len(objects)}")
        
    image_copy = np.copy(image.numpy_view())
    annotated_image = visualize(image_copy, detection_result)
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    cv2.imwrite("test.jpg", rgb_annotated_image)
    return FileResponse("test.jpg")


@app.post("/face_recognition")
async def face_recognition(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...)
):
    
    contents = await image1.read()
    filename1 = f"temp_{image1.filename}"
    with open(filename1, "wb") as f:
        f.write(contents)

    contents = await image2.read()
    filename2 = f"temp_{image2.filename}"
    with open(filename2, "wb") as f:
        f.write(contents)

    img1 = cv2.imread(filename1)
    img2 = cv2.imread(filename2)

    faces1 = fa_model.get(img1)
    assert len(faces1)==1

    faces2 = fa_model.get(img2)
    assert len(faces2)==1
        
    feats1 = np.array(faces1[0].normed_embedding, dtype=np.float32)
    feats2 = np.array(faces2[0].normed_embedding, dtype=np.float32)

    sims = np.dot(feats1, feats2.T)
    print(float(sims))

    """
    얼굴 인식 API
    
    두 개의 이미지를 업로드하여 얼굴 인식을 수행합니다.
    
    - **image1**: 첫 번째 이미지 파일
    - **image2**: 두 번째 이미지 파일
    
    Returns:
        JSON 응답
    """
    # 여기에 얼굴 인식 로직을 구현할 수 있습니다
    
    return {
            "message": "얼굴 인식 요청이 성공적으로 처리되었습니다",
            "similarity": float(sims)
        }

