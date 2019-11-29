# _*_ coding: utf-8 _*_
"""
@author: Jibao Wang
@time: 2019/11/28 13:50
"""

import flask
import os, pickle, getConfig, execute
from PIL import Image
import numpy as np
import pickle
from flask import request
import werkzeug


# 实例化一个Flask应用，命名为 imgClassifierWeb
app = flask.Flask("imgClassifierWeb")
config = getConfig.get_config(config_file="config.ini")

# 预测函数
@app.route("/predict/", endpoint='predict')
def cnn_predict():
    global secure_filename
    # 获取每个图像类别的名称
    filename = config['dataset_path'] + 'batches.meta'
    fp = open(filename, 'rb')
    label_name_dict = pickle.load(fp)['label_names']
    # 读取用户上传的图片
    img = Image.open(os.path.join(app.root_path, secure_filename))
    r, g, b = img.split()
    r_arr = np.array(r)
    g_arr = np.array(g)
    b_arr = np.array(b)
    image = np.concatenate((r_arr, g_arr, b_arr)).reshape((1, 32, 32, 3))/255
    predicted_class = label_name_dict[execute.predict(image)[0]]
    # 将返回的结果用页面渲染出来
    return flask.render_template('prediction_result.html', predicted_class=predicted_class)

# 上传图片
@app.route('/upload/', methods=["POST"])
def upload_image():
    global secure_filename
    if flask.request.method == "POST":
        img_file = flask.request.files["image_file"]
        # 生成一个安全的文件名
        secure_filename = werkzeug.secure_filename(img_file.filename)
        # 保存文件
        img_path = os.path.join(app.root_path, secure_filename)
        img_file.save(img_path)
        print("图片上传成功.")
        return flask.redirect(flask.url_for(endpoint='predict'))
    return "图片上传失败"

def redirect_upload():
    return flask.render_template(template_name_or_list="upload_image.html")

@app.route('/', endpoint='/')
def index():
    return redirect_upload()

app.run(host='0.0.0.0', port=12345, debug=True)