# _*_ coding: utf-8 _*_
"""
@author: Jibao Wang
@time: 2019/11/28 10:35
"""
# 执行器的实现，读取训练数据、实例化模型、模型训练、模型保存、调用预测

import tensorflow as tf
import numpy as np
from cnnModel import cnnModel
import os, pickle, time, getConfig, sys, random

config = getConfig.get_config(config_file='config.ini')

# 定义读取训练数据的函数
def read_data(data_path, im_dim, num_channels, num_files, images_per_file):
    # 获取文件夹中的文件名称
    filenames = os.listdir(data_path)
    # 创建一个空矩阵，用于保存训练图片的数据
    dataset_array = np.zeros(shape=(num_files*images_per_file, im_dim, im_dim, num_channels))
    dataset_labels = np.zeros(shape=(num_files*images_per_file), dtype=np.uint8)
    index = 0
    for filename in filenames:
        if filename[0:len(filename)-1] == 'data_batch_':
            print("正在处理数据文件：", filename)
            data_dict = unpickle_patch(data_path + filename)
            images_data = data_dict[b'data']
            print(images_data.shape)
            images_data_reshaped = np.reshape(images_data, newshape=(len(images_data), im_dim, im_dim, num_channels))
            dataset_array[index*images_per_file:(index+1)*images_per_file, :, :, :] = images_data_reshaped
            dataset_labels[index*images_per_file: (index+1)*images_per_file] = data_dict[b'labels']
            index = index + 1
    return dataset_array, dataset_labels

# 定义读取 pickle 格式文件的函数，pickle文件是一个二进制文件，读取为字典
def unpickle_patch(file):
    fp = open(file, 'rb')
    patch_dict = pickle.load(fp, encoding='bytes')
    return patch_dict

# 定义训练模型
def create_model():
    # 如果存在预训练模型，直接 load 并返回
    if 'pretrained_model' in config:
        model = tf.keras.models.load_model(config['pretrained_model'])
        return model

    ckpt = tf.io.gfile.listdir(config['working_directory'])
    if ckpt: # 非空
        model_file = os.path.join(config['working_directory'], ckpt[-1])
        print("Reading model parameters from %s" % model_file)
        model = tf.keras.models.load_model(model_file)
        return model
    else:
        model = cnnModel(config['droprate']).createModel()
        return model

def train(X, Y):
    model = create_model()
    history = model.fit(X, Y, verbose=2, epochs=30, validation_split=0.2)
    # 保存训练模型
    filename = 'cnn_model.h5'
    checkpoint_path = os.path.join(config['working_directory'], filename)
    model.save(checkpoint_path)

def predict(data):
    # ckpt = os.listdir(config['working_directory'])
    checkpoint_path = os.path.join(config['working_directory'], 'cnn_model.h5')
    model = tf.keras.models.load_model(checkpoint_path)
    prediction = model.predict(data)
    return tf.math.argmax(prediction, axis=1).numpy()

# 定义启动函数入口
if __name__ == '__main__':
    # 数据操作
    dataset_array, dataset_labels = read_data(data_path=config['dataset_path'], im_dim=config['im_dim'],
                                              num_channels=config['channels'], num_files=config['num_files'],
                                              images_per_file=config['images_per_file'])
    dataset_array = dataset_array.astype('float32') / 255  # 数据归一化
    dataset_labels = tf.keras.utils.to_categorical(dataset_labels, num_classes=config['num_dataset_classes'])
    if config['mode'] == 'train':
        train(dataset_array, dataset_labels)
    elif config['mode'] == 'server':
        print('请使用： python3 execute.py')
