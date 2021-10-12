#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""
import logging

def get_logger(logName, logFile=False):
    """To get a logger in one step.

    Logging is one of the most underrated features. Two things (5&3) to take away from 
    Logging in Python: 1) FIVE levels of importance that logs can contain(debug, info, warning, 
    error, critical); 2) THREE components to configure a logger in Python (a logger, a formatter, 
    and at least one handler).

    Args:
        logName: a logger name to display in loggings.
        logFile: a target file to save loggins.

    Return:
        logger: a well logger.
    """
    logger = logging.getLogger(logName)
    logger.setLevel(logging.DEBUG)
    
    formatter=logging.Formatter('%(asctime)s %(name)s | %(levelname)s -> %(message)s')
    
    if logFile:
        file_handler = logging.FileHandler(logFile)     # creating a handler to log on the filesystem
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()        # creating a handler to log on the console
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    
    return logger

def main():
    try:  
        logger = get_logger(logName='test', logFile='test.log')
        logger.info('this is a test')
        1/0
    except Exception as err:
        logger.error(err)
        raise err 

if __name__ == '__main__':
    main()