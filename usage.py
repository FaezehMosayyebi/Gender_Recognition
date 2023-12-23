import cv2
import tensorflow as tf
import numpy as np
from face_processes.face_detection import FaceDetection
from model import Trainer

class pipline:

    def __init__(self, model_dir)->None:
        
        self.model_dir = model_dir
        self.face_detector = FaceDetection()
        if model_dir is not None:
            self.model = tf.keras.models.load_model(model_dir)

    def train(self, data_dir:dict, model_name:str, batch_size:int, patch_size:(int, int), train_num_epochs:int, tune_num_epochs:int, tune_from:int, lr:float, aug_config:dict, save_to_dir):

        trainer = Trainer(data_dir['training'], data_dir['validation'], batch_size, patch_size)
        trainer.load_data
        trainer.model(model_name, aug_config)
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

        
    def run_model(self, img_dir):

        img = cv2.imread(img_dir)
        #cv2_imshow(img)
        #print(img.shape)
        face = self.face_detector(img)
        for i in range(len(face)):
            (x, y, x1, y1) = face[i].astype("int")
            image = img[y:y1, x:x1]
            image = cv2.resize(image, (224,224))
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