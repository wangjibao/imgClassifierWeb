# -*- coding: utf-8 -*-
"""
@author: spring
@time: 2019/11/27 14:14

"""
import configparser


def get_config(config_file='config.ini'):
    parser = configparser.ConfigParser()
    parser.read(config_file, encoding='utf-8')
    # 解析 ints, strings, floats 各个 section 的参数，并整成字典类型返回
    _conf_ints = [(key, int(value)) for key, value in parser.items('ints')]
    _conf_strings = [(key, str(value)) for key, value in parser.items('strings')]
    _conf_floats = [(key, float(value)) for key, value in parser.items('floats')]
    return dict(_conf_ints + _conf_strings + _conf_floats)


