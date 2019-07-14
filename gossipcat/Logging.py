#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""
import logging

def log_setup(log_file, log_name):
    """Logging is one of the most underrated features. Two things (5&3) to take away from 
    Logging in Python: 1) FIVE levels of importance that logs can contain(debug, info, warning, 
    error, critical); 2) THREE components to configure a logger in Python (a logger, a formatter, 
    and at least one handler).

    Args:
        log_file: a target file to save loggins.
        log_name: a logger name to display in loggings.
    Return:
        logger: a well setup logger.
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    
    formatter=logging.Formatter('%(asctime)s %(name)s | %(levelname)s -> %(message)s')
    
    file_handler = logging.FileHandler(log_file)    # creating a handler to log on the filesystem
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()        # creating a handler to log on the console
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    logger.addHandler(stream_handler)               # adding handlers to our logger
    logger.addHandler(file_handler)

    logger.info('happy logging')                    # test

    return logger

def main():
    try:  
        logger = log_setup(log_file='app.log', log_name='app')
        logger.info('test')
        1/0
    except Exception as err:
        logger.error(err)
        raise err 

if __name__ == '__main__':
    main()