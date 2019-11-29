# -*- coding: utf-8 -*-
"""
@author: spring
@time: 2019/11/27 15:06

"""

import tensorflow as tf
import numpy as np

# 定义 cnnModel 方法类，返回一个 CNN 实例
class cnnModel(object):
    def __init__(self, droprate):
        self.droprate = droprate

    def createModel(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), kernel_initializer='he_normal',
                                         strides=1, padding='same', activation='relu', input_shape=(32, 32, 3),
                                         name='conv1'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=1, padding='same', name='pool1'))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='he_normal',
                                         strides=1, padding='same', activation='relu', name='conv2'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=1, padding='same', name='pool2'))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer='he_normal',
                                         strides=1, padding='same', activation='relu', name='conv3'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=1, padding='same', name='pool3'))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Flatten(name='flatten'))
        model.add(tf.keras.layers.Dropout(rate=self.droprate, name='dropout'))
        model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

        model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=['accuracy'])
        return model
