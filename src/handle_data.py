import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data_path = 'dataset/smoke_detection_iot.csv'

class Data:
    def __init__(self , path = data_path):
        self.raw_data = pd.read_csv(path)

        # pick columns effect significantly
        self.effective_columns = ['UTC' , 'Temperature[C]' , 'Humidity[%]' ,
                                  'eCO2[ppm]', 'Raw H2' , 'Raw Ethanol' ,
                                  'Fire Alarm'
                                  ]

        self.x_train = self.x_test = None
        self.y_train = self.y_test = None

        self.train_ratio = 0.8

    def extract(self):
        # set up columns
        self.data = self.raw_data[self.effective_columns]

        # set UTC from 0 to 23
        self.data['UTC'] = pd.to_datetime(self.raw_data['UTC'] , unit='s')
        self.data['UTC'] = self.data['UTC'].dt.hour

    def split(self):
        x = self.data.drop(columns = ['Fire Alarm'])
        y = self.data['Fire Alarm']

        self.x_train , self.x_test , self.y_train , self.y_test = train_test_split(
            x , y , test_size= 1 - self.train_ratio , random_state=42
        )

    def scale(self):
        scaler = StandardScaler()
        self.x_train = scaler.fit_transform(self.x_train)
        self.x_test = scaler.fit_transform(self.x_test)

    def handleXY(self):
        self.extract()
        self.split()
        self.scale()
        return [self.x_train , self.x_test , self.y_train , self.y_test]

    # Utils
    def get_input_size(self):
        return len(self.effective_columns ) - 1

    def count_unique_values(self):
        for col in self.effective_columns :
            count = self.data[col].nunique()
            print(f'{col}: {count} values')



# print('start')
# data = Data()
# x_train = data.handleXY()[0]
# print(x_train[:5])
# print('done!!!')