import numpy as np
from PIL import Image, ImageDraw
import cv2
import os
from face_processes.facedetection.face_detection import FaceDetection
from face_processes.landmarkdetection.landmark_detection import LandmakDetector

class MaskGenerator:
    def __init__ (self, low_threshold, high_threshold, kernel_size):

        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.kernel_size = kernel_size
        self.face_detector = FaceDetection()
        self.landmark_detector = LandmakDetector()

    def mask_generator(self, img):

        rect = self.face_detector.detect(img)
        lm = self.landmark_detector.detect(img, rect)

        maskpoints = np.array([lm[0],lm[1],lm[2],
                                lm[3],lm[4],lm[5],
                                lm[6],lm[7],lm[8],
                                lm[9],lm[10],lm[11],
                                lm[12],lm[13],lm[14],
                                lm[15],lm[16],[(lm[14][0]+lm[28][0])/2, (lm[14][1]+lm[28][1])/2],
                                lm[28],[(lm[2][0]+lm[28][0])/2, (lm[2][1]+lm[28][1])/2]]  , dtype="float32")

        mask = Image.new('L', (img.shape[1], img.shape[0]), "black")
        ImageDraw.Draw(mask).polygon(maskpoints, outline="white", fill="white")
        mask = np.array(mask)
        masked_face = cv2.textureFlattening(img, mask, self.low_threshold, self.high_threshold, self.kernel_size)

        return masked_face
    
    def flow_from_directory(self, src_dir, dest_dir, prefix):

        for file in os.listdir("src_dir"): 
            if file.endswith(".jpg"): 

                img = cv2.imread(os.path.join(src_dir, file))

                if img is not None:
                    masked_image = self.mask_generator(img)
                    if prefix is not None:
                        file = prefix + file
                    cv2.imwrite(os.path.join(dest_dir, file), masked_image)


