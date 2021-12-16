import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Dense, ReLU, Dropout, Lambda
from typing import List

from block import ResNet



resnet = ResNet([64, 64], [5, 5], (256, 256))

resnet.build((256, 256))

print(resnet.summary())