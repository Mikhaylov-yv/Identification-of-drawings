import pandas as pd
import json
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
from keras.models import load_model
import numpy as np
import cv2
import os


class Train:
    def __init__(self):
        # Параметры
        # Размер изображения для обучения
        self.sithe = 512

        self.data_path_train = '../data/dataset/dataset/training_data'
        self.data_path_test = '../data/dataset/dataset/testing_data'
        self.test = False
        self.epochs = 6

    def train(self):
        self.train_df = self.get_annotation()
        self.train_filtr = self.__filtr(self.train_df)
        train_df = self.train_df[self.train_filtr]
        train_data, train_targets, train_filenames = self.get_img_data(train_df, self.data_path_train)
        data = np.array(train_data, dtype="float32") / 255.0
        targets = np.array(train_targets, dtype="float32")
        filenames = train_filenames
        split = train_test_split(data, targets, filenames, test_size=0.10,
                                 random_state=42)
        (trainImages, testImages) = split[:2]
        (trainTargets, testTargets) = split[2:4]
        vgg = VGG16(weights="imagenet", include_top=False,
                    input_tensor=Input(shape=(self.sithe, self.sithe, 3)))

        vgg.trainable = False
        flatten = vgg.output
        flatten = Flatten()(flatten)
        bboxHead = Dense(128, activation="relu")(flatten)
        bboxHead = Dense(64, activation="relu")(bboxHead)
        bboxHead = Dense(32, activation="relu")(bboxHead)
        bboxHead = Dense(4, activation="sigmoid")(bboxHead)
        model = Model(inputs=vgg.input, outputs=bboxHead)
        opt = Adam(
            learning_rate=1e-4
        )
        model.compile(loss="mse", optimizer=opt)
        model.fit(
            trainImages, trainTargets,
            validation_data=(testImages, testTargets),
            batch_size=3,
            epochs=self.epochs,
            verbose=1
        )
        if self.test != True: model.save('segm_model')


    def get_img_data(self, df, images_path):
        data = []
        targets = []
        filenames = []
        for name in df.fil_name.unique():
            filename = name + '.png'
            image_path = f'{images_path}/images/{filename}'
            image = cv2.imread(image_path)
            (h, w) = image.shape[:2]
            row = df.loc[df.fil_name == name, 'box'].max()
            (startX, startY, endX, endY) = row
            startX = float(startX) / w
            startY = float(startY) / h
            endX = float(endX) / w
            endY = float(endY) / h
            image = load_img(image_path, target_size=(self.sithe, self.sithe))
            image = img_to_array(image)
            data.append(image)
            targets.append((startX, startY, endX, endY))
            filenames.append(filename)
        return (data, targets, filenames)


    def get_annotation(self):
        annotations_path = self.data_path_train + '/annotations'
        df = pd.DataFrame()
        for fill_name in os.listdir(annotations_path):
            path = f"{annotations_path}/{fill_name}"
            name = fill_name.split('.')[0]
            f = open(path, encoding="utf-8")
            data = json.load(f)
            info_df = pd.json_normalize(data['form'])
            info_df['fil_name'] = name
            df = df.append(info_df)
        return df

    def __filtr(self, df):
        return (df.label == 'other')&(df.text.str.contains('(^\d{5,10})|(\d{5,10}$)'))