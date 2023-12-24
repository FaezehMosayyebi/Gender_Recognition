# author: Faezeh Mosayyebi
# This code is for detecting all faces in an image

import numpy as np
import dlib
import cv2
import os

class FaceDetection(object):
  def __init__(self) -> None:

    # Face detection model
    face_detection_model_prams = "res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "deploy.prototxt.txt"
    self.face_detection_model = cv2.dnn.readNetFromCaffe(configFile, face_detection_model_prams)

  def detect(self, image):

    h, w = image.shape[:2]
    rects = []

    blob = cv2.dnn.blobFromImage(image=cv2.resize(image, (300, 300)), scalefactor=1.0,
                                size=(300, 300), mean=(104.0, 117.0, 123.0))
    
    self.face_detection_model.setInput(blob)
    faces = self.face_detection_model.forward()
    for i in range(200):

      #confidence = faces[0, 0,  :, 2]

      if faces[0,0,i,2] > 0.5:
        box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x, y, x1, y1) = box.astype("int")

        if y<= image.shape[0] and y1<= image.shape[0] and x<= image.shape[1] and x1<=image.shape[1]:
          rects.append(dlib.rectangle(left=x, top=y, right=x1, bottom=y1))

    return rects
  
  def flow_from_directory(self, src_dir, dest_dir, prefix):

    for file in os.listdir("src_dir"): 
        if file.endswith(".jpg"): 
            
            img = cv2.imread(os.path.join(src_dir, file))

            if img is not None:
                rects = self.detect(img)
                for i in range(len(rects)):
                   cv2.rectangle(img, (rects[i].left(), rects[i].top()), (rects[i].right(), rects[i].bottom()), (255, 0, 0), 2)
                if prefix is not None:
                    file = prefix + file
                cv2.imwrite(os.path.join(dest_dir, file), img)