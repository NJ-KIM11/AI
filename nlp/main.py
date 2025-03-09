from typing import Union
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

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
    base_options = python.BaseOptions(model_asset_path='models\efficientnet_lite0.tflite')
    options = vision.ImageClassifierOptions(base_options=base_options, max_results=3)
    classifier = vision.ImageClassifier.create_from_options(options)
    image = mp.Image.create_from_file(image.file)
    classification_result = classifier.classify(image)
    top_category = classification_result.classifications[0].categories[0]
    print(f"{top_category.category_name} ({top_category.score:.2f})")
    
    return JSONResponse(
        content={
            "message": "이미지 분류 요청이 성공적으로 처리되었습니다",
            "image_filename": image.filename,
            "top_category": top_category.category_name,
            "score": top_category.score
        },
        status_code=200
    )

