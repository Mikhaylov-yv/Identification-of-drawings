import pandas as pd
import os
import json
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import PIL
import PIL.Image

class Train:
    def __init__(self):
        self.data_path_train = '../data/dataset/dataset/training_data'
        self.data_path_test = '../data/dataset/dataset/testing_data'

    def train(self):
        self.train_df = self.get_annotation()
        self.train_filtr = self.__filtr(self.train_df)

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