# text = "The Golden State Warriors are an American professional basketball team based in San Francisco."

from transformers import AutoTokenizer, BertForTokenClassification, logging
logging.set_verbosity_error()
# import sys, os, torch
# import numpy as np
# sys.path.insert(0, '../')
# import label
# import kss

from transformers import pipeline
text = "여야 원내대표가 16일 오후 김진표 국회의장 주재로 다시 얼굴을 맞대고 내년도 예산안 협상을 이어갔지만 기존 입장만 되풀이하며 진전을 보지 못했다. 이날 회동은 전날 김 의장이 내놓은 중재안을 국민의힘이 받아들이지 않으면서, 예산안 협상이 또 불발된 이후 첫 만남이었다. 양당 원내대표는 이날도 서로에게 '양보'를 요구하며 지루한 대치 국면을 이어갔다."

# classifier = pipeline("ner", model="stevhliu/my_awesome_wnut_model")
tokenizer = AutoTokenizer.from_pretrained("kpfbert")
# huggingface 개체명 인식 모델 불러오기
model = BertForTokenClassification.from_pretrained("KPF/KPF-bert-ner")
classifier = pipeline("ner", model)
result = classifier(text)
print(result)
