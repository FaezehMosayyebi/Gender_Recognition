import cv2
import tensorflow as tf
import numpy as np
from face_processes.facedetection.face_detection import FaceDetection
from face_processes.landmarkdetection.landmark_detection import LandmakDetector
from face_processes.maskgenerator import MaskGenerator
from model import Trainer
from modelevaluation import ModelEval
import os

class pipeline:

    def __init__(self, model_dir)->None:
        
        self.face_detector = FaceDetection()
        self.mask_generator = MaskGenerator()
        if model_dir is not None:
            self.model = tf.keras.models.load_model(model_dir)


    def train(self, data_dir:dict, model_name:str, batch_size:int, patch_size:(int, int), train_num_epochs:int, tune_num_epochs:int, tune_from:int, lr:float, augmentation:bool, aug_config:dict, save_to_dir:str):

        trainer = Trainer(data_dir['training'], data_dir['validation'], batch_size, patch_size)
        trainer.load_data()
        trainer.model(model_name, augmentation, aug_config)
        train_history, self.model = trainer.train(train_num_epochs, lr)
        trainer.plot_train_process(save_to_dir)
        if tune_num_epochs is not None:
            tune_history, self.model = trainer.fine_tune(tune_from, tune_num_epochs)
            trainer.plot_whole_process(save_to_dir)

        if save_to_dir is not None:
            trainer.save_model(save_to_dir)
            trainer.save_history(train_history, save_to_dir)
            if tune_num_epochs is not None:
                trainer.save_history(tune_history, save_to_dir)


    def evaluate_model(self, testdata_dir, batch_size, patch_size, confusion_matrix):

        evaluator = ModelEval(testdata_dir, batch_size, patch_size, self.model)
        evaluator.load_data()
        evaluator.evaluat()
        if confusion_matrix:
            evaluator.confusion_matrix()
        

    def run_model(self, img_dir, save_to_dir, save_path):

        img = cv2.imread(img_dir)
        face = self.face_detector.detect(img)
        for i in range(len(face)):
            x = face[i].left()
            y = face[i].top()
            x1 = face[i].right()
            y1 = face[i].bottom()
            image = img[y:y1, x:x1]
            image = cv2.resize(image, (240,240))
            image = tf.keras.applications.efficientnet.preprocess_input(image)
            image = tf.expand_dims(image, axis=0)
            predict= self.model.predict(image)
            predictions = tf.where(predict < 0.5, 0, 1)
            predictions = np.argmax(predictions, axis=1)
            if predictions==[1]:
                cv2.rectangle(img, (x, y), (x1, y1), (255, 0, 0), 2)
                cv2.putText(img, 'male', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,), 2)
            elif predictions==[0]:
                cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
                cv2.putText(img, 'Female', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            cv2.imshow(img)
            cv2.waitKey(0)
            if save_to_dir:
                cv2.imwrite(os.path.join(save_path, 'gender_detection_result.png'), img)

    def detect_faces(self, img_dir, save_to_dir, destination_directory, flow_from_dir, prefix):

        """
            Usage of face detection class
            Params:
                img_dir: Diretion of image of the diretory of images when flow_from_directory
                save_to_dir: do you want to save result?
                destination_directory: destination directory to save results.
                flow_from_directory: if you want to flow from directory
                prefix: the prefix you want to save files in flow_from_directory
        """
        if flow_from_dir:
            self.face_detector.flow_from_directory(img_dir, destination_directory, prefix)
        else:
            img = cv2.imread(img_dir)
            rects = self.face_detector.detect(img)

            for i in range(len(rects)):
                cv2.rectangle(img, (rects[i].left(), rects[i].top()), (rects[i].right(), rects[i].bottom()), (255, 0, 0), 2)

            if save_to_dir:
                if destination_directory is not None:
                    cv2.imwrite(os.path.join(destination_directory, 'face_detection_result.png'), img)
                else:
                    cv2.imwrite(img)

            cv2.imshow(img)
            cv2.waitKey(0)

    def detect_face_landmarks(self, image_dir, num_landmark, save_to_dir, destination_directory, flow_from_directory, prefix):

        """
            Usage of landmark detection class
            Params:
                image_dir: Diretion of image inf flow from directory the directory containing images
                num_landmark: the number of landmark that you want to be deteted (5 or 68)
                save_to_dir: do you want to save result?
                destination_directory: destination directory to save results.
                flow_from_directory: if you want to find the landmark of faces of images of a directory
                prefix: the prefix you want to save images with.
        """
        landmark_detector = LandmakDetector(num_landmark)

        if flow_from_directory:
            landmark_detector.flow_from_directory(image_dir, destination_directory, prefix)
        else:
            img = cv2.imread(image_dir)
            rects = self.face_detector.detect(img)
            
            for i in range(len(rects)):
                coords = landmark_detector.detect(img, rects[i])
                for (x, y) in coords:
                    cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), -1)

            if save_to_dir:
                if destination_directory is not None:
                    cv2.imwrite(os.path.join(destination_directory, 'landmark_detection_result.png'), img)
                else:
                    cv2.imwrite(img)


            cv2.imshow(img)
            cv2.waitKey(0)


    def generate_masked_faces(self, low_threshold:float, high_threshold:float, kernel_size:int, image_dir:str, save_to_dir:str, save_directory:str, flow_from_directory:bool, prefix:str):

        """
            Usage of masked face generator
            Params:
                low_threshold
                high_threshold
                kernel_size: The size of the Sobel kernel to be used
                image_dir: the image directory of the directory of images in flow from directory
                save_to_dir: do you want to save result?
                save_directory: destination directory to save results.
                flow_from_directory: whether you want to flow from directory or not
                prefix: the prefix you want to save you files with in flow from directory
        """

        if flow_from_directory:
            self.mask_generator.flow_from_directory(low_threshold, high_threshold, kernel_size, image_dir, save_directory, prefix)
        else:
            img = cv2.imread(image_dir)
            masked_face = self.mask_generator.mask_generator(img, low_threshold, high_threshold, kernel_size)
            if save_to_dir:
                if save_directory is not None:
                    cv2.imwrite(os.path.join(save_directory, 'landmark_detection_result.png'), img)
                else:
                    cv2.imwrite(img)


        



