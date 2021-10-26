#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""
import warnings
warnings.filterwarnings('ignore')

def getConfig(configFile):
    """To get configuration file in one step.

    Args:
        configFile: configuration file name, like 'config.json'.

    Returns:
        config: a dictionary contians configuration.
    """
    import json

    config = dict()
    try:
        with open(configFile, 'r') as f:
            config = json.load(f)
        return config 
    except Exception as e:
        print('[CRITIAL] NO CONFIGURATION FILE FOUND!')
        raise e

def install(package):
    """To install a Python package within Python.

    Args:
        package: package name, string.
    """
    import subprocess
    import sys

    try:
        subprocess.call([sys.executable, "-m", "pip", "install", package])
        print(package, ' successfuly installed.')
    except Exception as e:
        raise e

def flatten(df, feature_list, k_list):
    import ast
    for i, f in enumerate(feature_list):
        l = []
        for j in range(k_list[i]):
            l.append('{}_{}'.format(feature_list[i], j))
        df[feature_list[i]] = df[feature_list[i]].apply(lambda x: ast.literal_eval(x))
        df[l] = pd.DataFrame(df[feature_list[i]].tolist(), index=df.index)
        _ = df.pop(feature_list[i])
    return df