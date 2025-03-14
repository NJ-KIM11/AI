# STEP 1: import modules
import argparse
import cv2
import sys
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# STEP 2: create inference instance
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640,640))

# STEP 3: load data
img1 = cv2.imread('iu1.jpg')
img2 = cv2.imread('20160310000359_0.jpg')
# img = ins_get_image('t1')

# STEP 4: inference
faces1 = app.get(img1)
assert len(faces1)==1

faces2 = app.get(img2)
assert len(faces2)==1

# STEP 5: post processing

# 5-1: draw face bounding box
# rimg = app.draw_on(img, faces)
# cv2.imwrite("./t1_output.jpg", rimg)

# 5-2: calculate face similarity
# then print all-to-all face similarity

feats1 = np.array(faces1[0].normed_embedding, dtype=np.float32)
feats2 = np.array(faces2[0].normed_embedding, dtype=np.float32)

sims = np.dot(feats1, feats2.T)
print(sims)

