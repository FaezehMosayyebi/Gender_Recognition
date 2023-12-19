import numpy as np
import dlib
import cv2

class FaceDetection(object):
  def __init__(self) -> None:

    # Face detection model
    face_detection_model_prams = "res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "deploy.prototxt.txt"
    self.face_detection_model = cv2.dnn.readNetFromCaffe(configFile, face_detection_model_prams)

  def detect(self, image):

    h, w = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image=cv2.resize(image, (300, 300)), scalefactor=1.0,
                                size=(300, 300), mean=(104.0, 117.0, 123.0))
    
    self.face_detection_model.setInput(blob)
    faces = self.face_detection_model.forward()

    face = []
    confidence = faces[0, 0, 0, 2]

    if confidence > 0.5:
      box = faces[0, 0, 0, 3:7] * np.array([w, h, w, h])
      (x, y, x1, y1) = box.astype("int")

      if y<= image.shape[0] and y1<= image.shape[0] and x<= image.shape[1] and x1<=image.shape[1]:
        rect = dlib.rectangle(left=x, top=y, right=x1, bottom=y1)
      else:
        rect = dlib.rectangle(left=0, top=0, right=0, bottom=0)

    return rect