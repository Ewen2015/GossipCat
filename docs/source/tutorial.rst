GossipCat Tutorials
*******************

Data Science Projects Basics
============================

Environment Setup
-----------------

The first step to start a data science project should always be setup a development file system, no matter on cloud or in your laptop. **GossipCat** provides a one-line command to setup a well-organized file system for data science projects.

.. code-block:: bash

    python -m gossipcat.dev.FileSys

The interactive and immersive command-line interfaces as following. Just type down your project name, like :code:`battery` in this tutorial. Then it will generate a file structure for your data science project and print out a file tree of it. 

::

    hi, there! please write down your machine learning project's name.
    project's name: battery
    project_battery/
        requirements.txt
        README.md
        .gitignore
        docs/
            READM.md
        log/
        model/
        test/
        data/
            tmp/
            train/
            test/
            result/
            raw/
        notebook/
        report/
        script/
            config.json
        deploy/
            deploy.sh

.. note::

    1. :file:`requirements.txt` includes all packages you need in your project. We recommend you to list not only package names but thier versions in the file. Besides, this serves your well if you develop your project on SageMaker, for you have to install all required packages every time restarting the Jupyter Notebook instance.
    2. :file:`.gitignore` includes :file:`data/*` by default, which is our best practice in data science projects with **git**. Generally, you don't want to git your data. 
    3. :file:`docs/READM.md` is inspired by `How to ML Paper - A brief Guide <https://docs.google.com/document/d/16R1E2ExKUCP5SlXWHr-KzbVDx9DBUclra-EbU8IB-iE/edit?usp=sharing>`_. We highly recommend you to document your data science project in an organized way so that anyone, including youself, can catch up your thoughts in the future.


Logging
-------

Most data scientists spend little time on logging and may just print out along the experiement in Jupyter Notebook. However, this can make annoying troubles when it comes to production environment or when the data science experiements require a long period to generate experiement records. Therefore, logging is critical to a data science project. 

Python Module **Logging** is one of the most underrated features. Two things (5&3) to take away from **Logging**: 

1. **5 levels** of importance that logs can contain(debug, info, warning, error, critical);  
2. **3 components** to configure a logger in Python (a logger, a formatter, and at least one handler).

**GossipCat** provides a function :code:`get_logger` to make life easier.

.. code-block:: Python

    import gossipcat as gc
    
    log_name = 'battery'
    log_file = '../log/batter.log'

    logger = gc.get_logger(logName=log_name, logFile=log_file)
    
    logger.debug('this is a debug')
    logger.info('this is a test')
    logger.warning('this is a warning')
    
    logger.error('this is an error!')
    logger.critial('this is critical!')


Data Science Experiment
=======================


