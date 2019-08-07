#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""
import os 

def tree(path=os.getcwd()):
    """A function to draw a file system tree.
    """
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' '*4*(level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' '*4*(level+1)
        for f in files:
            print('{}{}'.format(subindent, f))
    return None

def FileSys():
    """To establish a machine learning project file system.
    """
    print("hi, there! please write down your machine learning project's name.")
    name = input("project's name: ")
    pn = 'project_'+name
    os.mkdir(pn)
    os.chdir(pn)

    run = 'run.sh'
    with open(run, 'a') as f:
        try:
            f.write("#!/bin/bash")
            os.chmod(run, 0o755)
            os.utime(run, None)
        except Exception as e:
            pass

    subdir = ['data', 'code', 'notebook', 'model', 'report', 'log']
    for d in subdir:
        os.mkdir(d)

    os.chdir('data')
    dir_data = ['raw', 'train', 'test', 'result', 'tmp']
    for d in dir_data:
        os.mkdir(d)
    os.chdir('../')

    os.chdir('code')
    config = 'config.json'
    with open(config, 'a') as f:
        try:
            f.write("{\"version\": \"\"}")
            os.utime(config, None)
        except Exception as e:
            pass 
    os.chdir('../')
    
    tree(os.getcwd())
    return None

def main():
    FileSys()

if __name__ == '__main__':
    main()











