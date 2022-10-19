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

Most problems in the industry are not crystal clear as data science or machine learning homework problems in school. Data scientists should work with other function teams closely to really understand the problem and try to figure out a practical way to solve it. AND it is not even necessary to be a data science or machine learning project –– **a data scientist is a problem solver first and can solve it with data science when necessary**.  

Even within data science, there are plenty of methods and algorithms to solve problems, which really depends on the **business** needs and **technique** feasibility. Also, this is where **creativity** happens. A good data scientist should be familiar with commonly used methods and able to pick up new methods if necessary to adapt to the needs both from business and technique.  

Leave the creative ones aside, **GossipCat** and this tutorial focus on commonly used methods, say classification and regression, to provide a quick start and to reduce repetitive work as much as possible. 

Framework Design
----------------

Target
~~~~~~

For supervised machine learning, it seems to be clear that you got labels (or target, dependent variables) in your data set. While the target definition does not always inherently exist. For example, 

1. Price predicting: listing price or selling price? 
2. Non-performing loans classification: A nonperforming loan (NPL) is a sum of borrowed money whose scheduled payments have not been made by the debtor for a period –– usually 90 or 180 days. So, 90 or 180? Any tolerance periods? 

**Other than the physical world, things are always defined by people and therefore can be very different from time to time and from scenario to scenario. Things are always changing in the physical world as well. Before talking about any concepts abstractly, define them concretely.** This is a teamwork involving both business and technique teams.   

Data scientists should always double confirm with business team about the target definition. Furthermore, data scientists should always be skeptical of the definition especially when the training results are too good to be true. Check if there is any **data leakage** in the definition.  

.. note::

    **Data Leakage**: Data leakage (or leakage) happens when your training data contains information about the target, but similar data will not be available when the model is used for prediction. This leads to high performance on the training set (and possibly even the validation data), but the model will perform poorly in production.

    In other words, leakage causes a model to look accurate until you start making decisions with the model, and then the model becomes very inaccurate.

    There are two main types of leakage: **target leakage** and **train-test contamination**.
    
    `More information here. <https://www.kaggle.com/code/alexisbcook/data-leakage>`_

Features
~~~~~~~~

Time Window
~~~~~~~~~~~


Experiemental Design
--------------------

Baseline
~~~~~~~~


Algorithm Comparison
~~~~~~~~~~~~~~~~~~~~


Cross Validation
~~~~~~~~~~~~~~~~


Hyper-parameter Tuning
~~~~~~~~~~~~~~~~~~~~~~


Error Analysis
~~~~~~~~~~~~~~


Explanation
~~~~~~~~~~~


Model Development
=================


Model development and maintenance is under the MLOps topic, which is a quite new but fast-growing area in the data science field. As it is out of the scope of GossipCat, we will not cover much content here. For more information, you may refer to Ewen’s another package `BatCat <https://batcat.readthedocs.io/>`_.

Git
----

Git is a version control system designed to track changes in a source code over time.

When many people work on the same project without a version control system it's total chaos. Resolving the eventual conflicts becomes impossible as none has kept track of their changes and it becomes very hard to merge them into a single central truth. Git and higher-level services built on top of it (like Github) offer tools to overcome this problem.

Docker
------

Docker is a software container platform that provides an isolated container for us to have everything we need for our experiments to run. 

Essentially, it is a light-weight Virtual Machine (VM) built from a script that can be version controlled; so we can now version control our data science environment! Developers use Docker when collaborating on code with coworkers and they also use it to build agile software delivery pipelines to ship new features faster.