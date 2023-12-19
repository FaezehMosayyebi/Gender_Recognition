import numpy as np
import cv2
from PIL import Image, ImageDraw
from face_processes.landmark_detection import LandmakDetector
from face_align import FaceAlign

class FaceComponentsExtractor(object):
  def __init__(self, image) -> None:
    landmark_detector = LandmakDetector(64)
    face_aligner = FaceAlign(detect_face=True)

    self.aligned_face = face_aligner.align(image)
    self.landmarks =  landmark_detector.detect(self.aligned_face)


    self.facetop      = self.aligned_face.top()
    self.facebottom   = self.aligned_face.bottom()
    self.faceleft     = self.aligned_face.left()
    self.faceright    = self.aligned_face.right()
    self.eybrowtop    = int(min( self.landmarks[17][1],  self.landmarks[18][1],  self.landmarks[19][1],  self.landmarks[20][1],  self.landmarks[21][1], self.landmarks[22][1],  self.landmarks[23][1],  self.landmarks[24][1],  self.landmarks[25][1],  self.landmarks[26][1]))
    self.eyebrowleft  = int( self.landmarks[17][0])
    self.eyebrowright = int( self.landmarks[26][0])
    self.lipsbottom   = int(max( self.landmarks[55][1],  self.landmarks[56][1],  self.landmarks[57][1],  self.landmarks[58][1],  self.landmarks[59][1]))
    self.eyebottom    = int(max( self.landmarks[40][1],  self.landmarks[41][1],  self.landmarks[46][1],  self.landmarks[47][1]))
    self.righteyeleft =  self.landmarks[42]
    self.lefteyeright =  self.landmarks[39]
    self.nosetip      =  self.landmarks[30]

  def whole_image(self):
    """ 
      Whole face
    """
    y1m= self.facetop
    y2m= self.facebottom
    x1m= self.faceleft
    x2m= self.faceright
    return self.aligned_face[y1m:y2m , x1m:x2m]
  
  def all_face_components(self):
    """
      All face components
    """
    return self.aligned_face[self.eybrowtop:self.lipsbottom , self.eyebrowleft:self.eyebrowright]
  
def lower_part1(self):
  """
    From bottom of eyes to bottom of face
  """
  return self.aligned_face[self.eyebottom:self.facebottom , self.faceleft:self.faceright]

def lower_part2(self):
  """
    From top of eybrowa to bottom of face
  """
  return self.aligned_face[self.eybrowtop:self.facebottom , self.faceleft:self.faceright]

#top parts of face
def top_part1(self):
  """
    From top of face to nose tip
  """
  return self.aligned_face[self.aligned_face.top():self.nosetip[1] , self.faceleft:self.faceright]

def top_part2(self):
  """
    From face top to nose tip. From left of face to left corner of right eye.
  """
  return self.aligned_face[self.aligned_face.top():self.nosetip[1] , self.faceleft:self.righteyeleft[0]]

def top_part3(self):
  """
    From top of nose tip: From right corner of left eye to right of face.
  """
  return self.aligned_face[self.aligned_face.top():self.nosetip[1] , self.lefteyeright[0]:self.faceright]

def right_part1(self):
  """
    From nose tip to right of face.
  """
  return self.aligned_face[self.aligned_face.top():self.facebottom , self.nosetip[0]:self.faceright]

def right_part2(self):
  """
    From right corner of left eye to right of face.
  """
  return self.aligned_face[self.aligned_face.top():self.facebottom , self.lefteyeright[0]:self.faceright]


def right_part3(self):
  """
    Bottom right of face
  """
  return  self.aligned_face[self.eybrowtop:self.facebottom , int(min(self.landmarks[38][0],self.landmarks[40][0])):self.faceright]

def left_part1(self):
  """
    From nose tip to face left.
  """
  return self.aligned_face[self.aligned_face.top():self.facebottom , self.faceleft:self.nosetip[0]]

def left_part2(self):
  """
    From left corner of right eye to left of face.
  """
  return self.aligned_face[self.aligned_face.top():self.facebottom , self.faceleft:self.righteyeleft[0]]


def left_part3(self):
  """
    Bottom left of face
  """
  return self.aligned_face[self.eybrowtop:self.facebottom , self.faceleft:int(max(self.landmarks[43][0],self.landmarks[47][0]))]


def face_boundary(self):
  y1m= self.facetop
  y2m= self.facebottom
  x1m= self.faceleft
  x2m= self.faceright
  maskpoints = np.array([self.landmarks[0],self.landmarks[1],self.landmarks[2],
                         self.landmarks[3],self.landmarks[4],self.landmarks[5],
                         self.landmarks[6],self.landmarks[7],self.landmarks[8],
                         self.landmarks[9],self.landmarks[10],self.landmarks[11],
                         self.landmarks[12],self.landmarks[13],self.landmarks[14],
                         self.landmarks[15],self.landmarks[16],self.landmarks[26],
                         self.landmarks[25],self.landmarks[24],self.landmarks[23],
                         self.landmarks[22],self.landmarks[21],self.landmarks[20],
                         self.landmarks[19],self.landmarks[18],self.landmarks[17]]  , dtype="float32")
  mask = Image.new('L', (self.landmarks.shape[1], self.landmarks.shape[0]), "black")
  ImageDraw.Draw(mask).polygon(maskpoints, outline="white", fill="white")
  mask = np.array(mask)
  masked = cv2.textureFlattening(self.aligned_face, mask, 100, 2000, 50)
  return masked[y1m:y2m , x1m:x2m]