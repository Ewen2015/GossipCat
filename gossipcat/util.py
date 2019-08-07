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