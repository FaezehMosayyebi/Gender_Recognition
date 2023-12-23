import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from mlxtend.plotting import plot_confusion_matrix
from skimage.io import imread_collection

import numpy as np
import cv2


class ModelEval:
    
    def __init__(self, testdata_dir, batch_size, patch_size, model):

        self.test_data_dir = testdata_dir
        if type(model) == str:
            self.model_dir = model
        else:
            self.model = model
        self.batch_size = batch_size
        self.patch_size = patch_size

    def load_data(self):
        
        try:
            self.test_dataset = image_dataset_from_directory(self.test_data_dir,
                                                label_mode="categorical",
                                                shuffle=True,
                                                batch_size=self.batch_size,
                                                image_size=self.patch_size)
        
            class_names = self.test_dataset.class_names

            print(f'Data loaded successfully. Classes: {class_names}')

        except Exception as e:
            print('Data loading was unsuccessful. Check the following error')
            print(f'Error: {e}')
            exit(0)

    def load_model(self):
        self.model = tf.keras.models.load_model(self.model_dir)

    def evaluat(self):
        loss, accuracy = self.model.evaluate(self.test_dataset)

        print(f'Model evaluated successfully.')
        print(f'Test accuracy = {accuracy}')
        print(f'Test loss = {loss}')

    def confusion_matrix(self):

        male_data = imread_collection(self.test_data_dir +'/male/*.jpg')
        female_data = imread_collection(self.test_data_dir +'/female/*.jpg')

        true_male = 0
        true_femal = 0

        for i in range (len(male_data)):
        
            img = cv2.resize(male_data[i], (224,224))
            img = tf.keras.applications.efficientnet.preprocess_input(img)
            img = tf.expand_dims(img, axis=0)

            predictions = self.model.predict(img)
            predictions = tf.where(predictions < 0.5, 0, 1)                       
            predictions = np.argmax(predictions, axis=1)

            if predictions==[1]:
                true_male += 1

        for i in range (len(female_data)):
        
            img = cv2.resize(female_data[i], (224,224))
            img = tf.keras.applications.efficientnet.preprocess_input(img)
            img = tf.expand_dims(img, axis=0)

            predictions = self.model.predict(img)
            predictions = tf.where(predictions < 0.5, 0, 1)                       
            predictions = np.argmax(predictions, axis=1)

            if predictions==[0]:
                true_female += 1


        false_male = len(male_data) - true_male
        false_female = len(female_data) - true_female

        

        # plotting Confusion Matrix
        cm = np.array([[true_male, false_male],
                    [false_female, true_female]])

        # Classes
        classes = ['Male', 'Female']

        figure, ax = plot_confusion_matrix(conf_mat = cm,
                                        class_names = classes,
                                        show_absolute = False,
                                        show_normed = True,
                                        colorbar = True)
