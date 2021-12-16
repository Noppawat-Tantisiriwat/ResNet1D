import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Conv1D, BatchNormalization, ReLU, Add, GlobalAveragePooling1D, Activation, Dense
from typing import List

class ResidualUnit1D(Model):

    def __init__(self, filters :List, kernels : List,padding="same",*args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv1 = Conv1D(filters=filters[0], kernel_size=kernels[0],padding=padding)
        self.conv2 = Conv1D(filters=filters[1], kernel_size=kernels[0],padding=padding)
        self.activation = ReLU()
        self.batchnorm = BatchNormalization(axis=-1)


    def call(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.batchnorm(x)



class ResNet(Model):

    def __init__(self, filters : List, kernels : List, units=18,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units
        self.res_unit = ResidualUnit1D(filters=filters, kernels=kernels)
        self.add = Add()
        self.activation = ReLU()
        self.pooling = GlobalAveragePooling1D()

        self.dense = Dense(128)
        self.out = Dense(1)
        self.logits = Activation("sigmoid")
    

    
    def call(self, x):

        for layer in range(self.units):
            x_skip = x
            x = self.res_unit(x)
            x = self.add([x, x_skip])
            x = self.activation(x)

        x = self.pooling(x)
        x = self.dense(x)
        x = self.activation(x)
        x = self.out(x)
        x = self.logits(x)
        return x





        