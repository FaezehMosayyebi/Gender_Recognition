# author: Faezeh Mosayyebi
# this code generates masked faces.

import numpy as np
from PIL import Image, ImageDraw
import cv2
import os
from face_processes.facedetection.face_detection import FaceDetection
from face_processes.landmarkdetection.landmark_detection import LandmakDetector

class MaskGenerator:
    def __init__ (self):

        self.face_detector = FaceDetection()
        self.landmark_detector = LandmakDetector(68)

    def mask_generator(self, img, low_threshold, high_threshold, kernel_size):

        rects = self.face_detector.detect(img)

        for i in range(len(rects)):

            lm = self.landmark_detector.detect(img, rects[i])

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
            img = cv2.textureFlattening(img, mask, low_threshold, high_threshold, kernel_size)

        return img
    
    def flow_from_directory(self, low_threshold, high_threshold, kernel_size, src_dir, dest_dir, prefix):

        for file in os.listdir("src_dir"): 
            if file.endswith(".jpg"): 

                img = cv2.imread(os.path.join(src_dir, file))

                if img is not None:
                    masked_image = self.mask_generator(img, low_threshold, high_threshold, kernel_size)
                    if prefix is not None:
                        file = prefix + file
                    cv2.imwrite(os.path.join(dest_dir, file), masked_image)