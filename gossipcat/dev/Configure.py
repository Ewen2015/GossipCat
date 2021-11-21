#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""
import warnings
warnings.filterwarnings('ignore')

import json

def Configure(addin=None):
    """A configuration method for a machine learning project.
    
    Read configurations from config.json under the same directory.
    The config.json must include an item 'version' to speicify the 
    data version, which is used for the machine learning project.
    """
    config = dict()
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except Exception as e:
        print('[CRITIAL] NO CONFIGURATION FILE FOUND!')
        raise e

    system = {"wd_log": "../log/",
              "wd_raw": "../data/raw/",
              "wd_train": "../data/train/",
              "wd_test": "../data/test/",
              "wd_result": "../data/result/",
              "wd_report": "../report/",
              "wd_model": "../model/",

              "project_log": "project.log"}
    config = {**config, **system}

    try:
        config["file_log"] = "log_"+config["version"]+".log"
        config["file_raw"] = "raw_"+config["version"]+".csv"
        config["file_train"] = "train_"+config["version"]+".csv"
        config["file_test"] = "test_"+config["version"]+".csv"
    except Exception as e:
        pass

    if addin == None:
        pass
    else:
        try:
            config = {**config, **addin}
        except Exception as e:
            pass   
    return config

def getConfig():
    config = dict()
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        return config 
    except Exception as e:
        print('[CRITIAL] NO CONFIGURATION FILE FOUND!')
        raise e