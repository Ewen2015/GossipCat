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

def FileSys(project_name=True):
    """Establish a data science project file system.

    Args:
        project_name (bool): If a project name is needed, default True.

    Yields:
        A data science project file system. 
    """
    if project_name:
        print("hi, there! please write down your data science project's name.")
        name = input("project's name: ")
        pn = 'project_'+name
        os.mkdir(pn)
        os.chdir(pn)

        readme = 'README.md'
        with open(readme, 'a') as f:
            try:
                f.write("# {}".format(name))
                os.utime(readme, None)
            except Exception as e:
                pass 

    subdir = ['docs', 'data', 'notebook', 'script', 'deploy', 'model', 'test', 'report', 'log']
    for d in subdir:
        os.mkdir(d)
    
    with open('requirements.txt', 'a') as f:
        try:
            f.write("pandas==1.3.4")
            f.write("matplotlib==3.3.2")
            f.write("scikit-learn==0.24.2")
            os.utime(deploy, None)
        except Exception as e:
            pass

    with open('.gitignore', 'a') as f:
        try:
            f.write("data/*")
            os.utime(deploy, None)
        except Exception as e:
            pass

    os.chdir('deploy')
    deploy = 'deploy.sh'
    with open(deploy, 'a') as f:
        try:
            f.write("#!/bin/bash")
            os.chmod(deploy, 0o755)
            os.utime(deploy, None)
        except Exception as e:
            pass
    os.chdir('../')

    os.chdir('script')
    config = 'config.json'
    with open(config, 'a') as f:
        try:
            f.write("{\"version\": \"\"}")
            os.utime(config, None)
        except Exception as e:
            pass 
    os.chdir('../')

    os.chdir('data')
    dir_data = ['raw', 'train', 'test', 'result', 'tmp']
    for d in dir_data:
        os.mkdir(d)
    os.chdir('../')

    paper = \
"""[original document](https://docs.google.com/document/d/16R1E2ExKUCP5SlXWHr-KzbVDx9DBUclra-EbU8IB-iE/edit?usp=sharing)

# How to ML Paper - A brief Guide

Feel free to comment / share and happy paper writing! Also, please see caveats* below. 

## Canonical ML Paper Structure

### Abstract (TL;DR of paper):
- X: What are we trying to do and why is it relevant?
- Y: Why is this hard? 
- Z: How do we solve it (i.e. our contribution!)
- 1: How do we verify that we solved it:
    - 1a) Experiments and results
    - 1b) Theory 

### Introduction (Longer version of the Abstract, i.e. of the entire paper):
- X: What are we trying to do and why is it relevant?
- Y: Why is this hard? 
- Z: How do we solve it (i.e. our contribution!)
- 1: How do we verify that we solved it:
    - 1a) Experiments and results
    - 1b) Theory 
- 2: New trend: specifically list your contributions as bullet points (credits to Brendan)
- Extra space? Future work!
- Extra points for having Figure 1 on the first page

### Related Work:
- Academic siblings of our work, i.e. alternative attempts in literature at trying to solve the same problem. 
- Goal is to “Compare and contrast” - how does their approach differ in either assumptions or method? If their method is applicable to our problem setting I expect a comparison in the experimental section. If not there needs to be a clear statement why a given method is not applicable.   
- Note: Just describing what another paper is doing is not enough. We need to compare and contrast.

### Background:
- Academic Ancestors of our work, i.e. all concepts and prior work that are required for understanding our method. 
- Includes a subsection Problem Setting which formally introduces the problem setting and notation (Formalism) for our method. Highlights any specific assumptions that are made that are unusual. 

### Method:
- What we do. Why we do it. All described using the general Formalism introduced in the Problem Setting and building on top of the concepts / foundations introduced in Background.

### Experimental Setup:
- How do we test that our stuff works? Introduces a specific instantiation of the Problem Setting and specific implementation details of our Method for this Problem Setting. 

### Results and Discussion:
- Shows the results of running Method on our problem described in Experimental Setup. Compares to baselines mentioned in Related Work. Includes statistics and confidence intervals. Includes statements on hyperparameters and other potential issues of fairness. Includes ablation studies to show that specific parts of the method are relevant. Discusses limitations of the method. 

### Conclusion:
- We did it. This paper rocks and you are lucky to have read it (i.e. brief recap of the entire paper). Also, we’ll do all these other amazing things in the future. 
- To keep going with the analogy, you can think of future work as (potential) academic offspring (credits to James)."""
    
    os.chdir('docs')
    with open('READM.md', 'a') as f:
        try:
            f.write(paper)
        except Exception as e:
            pass
    os.chdir('../')
    
    tree(os.getcwd())
    return None

def main():
    FileSys()

if __name__ == '__main__':
    main()