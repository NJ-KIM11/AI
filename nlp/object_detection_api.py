from typing import Union
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
import time
import cv2 as cv2
import numpy as np

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


# MediaPipe 모델 설정
# base_options = python.BaseOptions(model_asset_path='models/efficientnet_lite0.tflite')
# options = vision.ImageClassifierOptions(base_options=base_options, max_results=3)
# classifier = vision.ImageClassifier.create_from_options(options)

base_options = python.BaseOptions(model_asset_path='models\efficientdet_lite0.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

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

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/obj_det")
async def obj_det(
    image: UploadFile = File(...)
):
    # 이미지 파일 저장
    contents = await image.read()
    filename = f"temp_{image.filename}"
    with open(filename, "wb") as f:
        f.write(contents)

    mp_image = mp.Image.create_from_file(filename)
    detection_result = detector.detect(mp_image)

# STEP 5: Process the detection result. In this case, visualize it.

    objects = []
    for detection in detection_result.detections:
        if detection.categories[0].category_name == 'person':
            objects.append(detection)
        print(f"Find Person : {len(objects)}")

    image_copy = np.copy(image.numpy_view())
    annotated_image = visualize(image_copy, detection_result)
    # bgr을 rgb로 변환
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    # cv2.imshow("test", rgb_annotated_image)
    cv2.imwrite("test.jpg", rgb_annotated_image)

    return FileResponse("test.jpg")

