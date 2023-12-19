from numpy import np
from PIL import Image
from face_processes.face_detection import FaceDetection
from face_processes.landmark_detection import LandmakDetector

class FaceAlign(object):
  def __init__(self, detect_face=True) -> None:

    if detect_face:
      self.face_detector = FaceDetection()
    self.landmark_detector = LandmakDetector(5)

  def align(self, image):

    face_rect = self.face_detector.detect(image)
    landmarks = self.landmark_detector.detect(image, face_rect)

    nose = landmarks[4]
    left_eye_x = int(landmarks[3][0] + landmarks[2][0]) // 2
    left_eye_y = int(landmarks[3][1] + landmarks[2][1]) // 2
    right_eye_x = int(landmarks[1][0] + landmarks[0][0]) // 2
    right_eye_y = int(landmarks[1][1] + landmarks[0][1]) // 2

    center_of_forehead = ((left_eye_x + right_eye_x) // 2, (left_eye_y + right_eye_y) // 2)
    center_pred = (int((face_rect.left() + face_rect.right()) / 2), int((face_rect.top() + face_rect.top()) / 2))

    length_line1 = np.sqrt((center_of_forehead[0] - nose[0]) ** 2 + (center_of_forehead[1] - nose[1]) ** 2)
    length_line2 = np.sqrt((center_pred[0] - nose[0]) ** 2 + (center_pred[1] - nose[1]) ** 2)
    length_line3 = np.sqrt((center_pred[0] - center_of_forehead[0]) ** 2 + (center_pred[1] - center_of_forehead[1]) ** 2)

    cos_a = -(length_line3 ** 2 - length_line2 ** 2 - length_line1 ** 2) / (2 * length_line2 * length_line1)
    angle = np.arccos(cos_a)

    qx = nose[0] + np.cos(angle) * (center_of_forehead[0] - nose[0]) - np.sin(angle) * (center_of_forehead[1] - nose[1])
    qy = nose[1] + np.sin(angle) * (center_of_forehead[0] - nose[0]) + np.cos(angle) * (center_of_forehead[1] - nose[1])

    if qx == qx or qy == qy:

      rotated_point = (int(qx), int(qy))

      c1 = (center_of_forehead[0] - nose[0]) * (rotated_point[1] - nose[1]) - (center_of_forehead[1] - nose[1]) * (rotated_point[0] - nose[0])
      c2 = (center_pred[0] - center_of_forehead[0]) * (rotated_point[1] - center_of_forehead[1]) - (center_pred[1] - center_of_forehead[1]) * (rotated_point[0] - center_of_forehead[0])
      c3 = (nose[0] - center_pred[0]) * (rotated_point[1] - center_pred[1]) - (nose[1] - center_pred[1]) * (rotated_point[0] - center_pred[0])

      if (c1 < 0 and c2 < 0 and c3 < 0) or (c1 > 0 and c2 > 0 and c3 > 0):
        angle = np.degrees(-angle)
      else:
        angle = np.degrees(angle)

      image = Image.fromarray(image)
      image = np.array(image.rotate(angle))


    return image