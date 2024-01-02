from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import pandas as pd
import os
from utils import *


class Trainer:
    def __init__(
        self,
        train_data_dir: str,
        validation_data_dir: str,
        batch_size: int,
        img_size: tuple[int, int],
    ) -> None:
        if path_valiadtor(train_data_dir) and path_valiadtor(validation_data_dir):
            self.train_dir = train_data_dir
            self.valid_dir = validation_data_dir

        if batch_size != 0:
            self.batch_size = batch_size
        else:
            batch_size = 16

        if img_size[0] != 0 and img_size[1] != 0:
            self.img_size = img_size
        else:
            self.img_size = (240, 240)

    def load_data(self) -> None:
        try:
            self.train_dataset = image_dataset_from_directory(
                self.train_dir,
                label_mode="categorical",
                shuffle=True,
                batch_size=self.batch_size,
                image_size=self.img_size,
            )

            self.validation_dataset = image_dataset_from_directory(
                self.valid_dir,
                label_mode="categorical",
                shuffle=True,
                batch_size=self.batch_size,
                image_size=self.img_size,
            )

            class_names = self.validation_dataset.class_names

            self.train_dataset = self.train_dataset.prefetch(
                buffer_size=tf.data.AUTOTUNE
            )
            self.validation_dataset = self.validation_dataset.prefetch(
                buffer_size=tf.data.AUTOTUNE
            )

            print(f"Data loaded successfully. Classes: {class_names}")

        except Exception as e:
            print("Data loading was unsuccessful. Check the following error")
            print(f"Error: {e}")
            exit(0)

    def data_augmentation(
        self,
        rotation_factor: float,
        translation_factor: tuple[float, float],
        flip: str,
        contrast_factor: float,
    ):
        data_augmentation = tf.keras.Sequential()

        if rotation_factor is not None:
            data_augmentation.add(
                tf.keras.layers.experimental.preprocessing.RandomRotation(
                    factor=rotation_factor
                )
            )

        if translation_factor is not None:
            data_augmentation.add(
                tf.keras.layers.experimental.preprocessing.RandomTranslation(
                    height_factor=translation_factor[0],
                    width_factor=translation_factor[1],
                )
            )

        if flip is not None:
            data_augmentation.add(
                tf.keras.layers.experimental.preprocessing.RandomFlip(mode=flip)
            )

        if contrast_factor is not None:
            data_augmentation.add(
                tf.keras.layers.experimental.preprocessing.RandomContrast(
                    factor=contrast_factor
                )
            )

        return data_augmentation

    def augment_plotter(
        self, aug_config: dict[float, tuple[float, float], str, tuple[float, float]]
    ) -> None:
        for image, _ in self.train_dataset.take(1):
            plt.figure(figsize=(10, 10))
            first_image = image[0]
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                augmented_image = self.data_augmentation(
                    aug_config["rotation_factor"],
                    aug_config["translation_factor"],
                    aug_config["flip"],
                    aug_config["contrast_factor"],
                )(tf.expand_dims(first_image, 0))
                plt.imshow(augmented_image[0] / 255)
                plt.axis("off")

    def model(
        self,
        model_size: str,
        augmentation: bool,
        aug_config: dict[float, tuple[float, float], str, tuple[float, float]],
    ) -> None:
        self.base_model = tf.keras.applications.EfficientNetB1(
            include_top=False, weights="imagenet", input_shape=self.img_size + (3,)
        )

        self.base_model.trainable = False

        if augmentation:
            da = self.data_augmentation(
                aug_config["rotation_factor"],
                aug_config["translation_factor"],
                aug_config["flip"],
                aug_config["contrast_factor"],
            )
        preprocess_input = tf.keras.applications.efficientnet.preprocess_input

        # model
        inputs = tf.keras.Input(shape=self.img_size + (3,))
        if augmentation:
            x = da(inputs)
            x = preprocess_input(x)
        else:
            x = preprocess_input(inputs)
        x = self.base_model(x, training=False)

        # top model

        x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        if model_size == "Large":
            x = tf.keras.layers.Dropout(0.2)(x)
            x = tf.keras.layers.Dense(2048, activation="relu")(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            x = tf.keras.layers.Dense(1024, activation="relu")(x)

        elif model_size == "Small":
            x = tf.keras.layers.Dropout(0.2)(x)
            x = tf.keras.layers.Dense(256, activation="relu")(x)

        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(2, activation="softmax")(x)

        self.model = tf.keras.Model(inputs, outputs)

    def train(self, num_epochs: int, learning_rate: float):
        self.train_num_epochs = num_epochs
        self.learning_rate = learning_rate

        self.model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            metrics=["accuracy"],
        )

        self.train_history = self.model.fit(
            self.train_dataset,
            epochs=self.train_num_epochs,
            validation_data=self.validation_dataset,
        )

        return self.train_history, self.model

    def fine_tune(self, tune_from: int, num_epochs: int):
        self.base_model.trainable = True
        num_layers = len(self.base_model.layers)

        for layer in self.base_model.layers[:tune_from]:
            layer.trainable = False

        print(f"{num_layers-tune_from} out of {num_layers} layers will be tuened.")

        self.model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.RMSprop(
                learning_rate=self.learning_rate / 10
            ),
            metrics=["accuracy"],
        )

        self.history_fine = self.model.fit(
            self.train_dataset,
            epochs=self.train_num_epochs + num_epochs,
            initial_epoch=self.train_history.epoch[-1],
            validation_data=self.validation_dataset,
        )

        return self.history_fine, self.model

    def plot_train_process(self, save_to_dir: str) -> None:
        directory_maker(save_to_dir)

        acc = self.train_history.history["accuracy"]
        val_acc = self.train_history.history["val_accuracy"]

        loss = self.train_history.history["loss"]
        val_loss = self.train_history.history["val_loss"]

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label="Training Accuracy")
        plt.plot(val_acc, label="Validation Accuracy")
        plt.legend(loc="lower right")
        plt.ylabel("Accuracy")
        plt.ylim([min(plt.ylim()), 1])
        plt.title("Training and Validation Accuracy")

        plt.subplot(2, 1, 2)
        plt.plot(loss, label="Training Loss")
        plt.plot(val_loss, label="Validation Loss")
        plt.legend(loc="upper right")
        plt.ylabel("Cross Entropy")
        plt.ylim([0, 1.0])
        plt.title("Training and Validation Loss")
        plt.xlabel("epoch")
        plt.show()
        if save_to_dir is not None:
            plt.savefig(os.path.join(save_to_dir, "train_plots.png"))

    def plot_whole_process(self, save_to: str) -> None:
        directory_maker(save_to)

        acc = (
            self.train_history.history["accuracy"]
            + self.history_fine.history["accuracy"]
        )
        val_acc = (
            self.train_history.history["val_accuracy"]
            + self.history_fine.history["val_accuracy"]
        )

        loss = self.train_history.history["loss"] + self.history_fine.history["loss"]
        val_loss = (
            self.train_history.history["val_loss"]
            + self.history_fine.history["val_loss"]
        )

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label="Training Accuracy")
        plt.plot(val_acc, label="Validation Accuracy")
        plt.ylim([0.8, 1])
        plt.plot(
            [self.train_num_epochs - 1, self.train_num_epochs - 1],
            plt.ylim(),
            label="Start Fine Tuning",
        )
        plt.legend(loc="lower right")
        plt.title("Training and Validation Accuracy")

        plt.subplot(2, 1, 2)
        plt.plot(loss, label="Training Loss")
        plt.plot(val_loss, label="Validation Loss")
        plt.ylim([0, 1.0])
        plt.plot(
            [self.train_num_epochs - 1, self.train_num_epochs - 1],
            plt.ylim(),
            label="Start Fine Tuning",
        )
        plt.legend(loc="upper right")
        plt.title("Training and Validation Loss")
        plt.xlabel("epoch")
        plt.show()
        if save_to is not None:
            plt.savefig(os.path.join(save_to, "whol_process_plots.png"))

    def save_model(self, save_to: str) -> None:
        directory_maker(save_to)
        self.model.save(os.path.join(save_to, "genderdetector.h5"))

    def save_history(self, history, save_to: str) -> None:
        directory_maker(save_to)

        # convert the history.history dict to a pandas DataFrame:
        hist_df = pd.DataFrame(history.history)

        # save to csv:
        with open(os.path.join(save_to, "history_fine.csv"), mode="w") as f:
            hist_df.to_csv(f)
