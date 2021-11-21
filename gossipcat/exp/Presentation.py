#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""

import nbformat as nbf 
        
def setup():
    
    nb = nbf.v4.new_notebook()

    title = """\
    # 数据科学咨询项目演示稿

    **作者：**

    **时间：**

    根据(一周数据科学咨询框架)[https://www.linkedin.com/pulse/one-week-data-science-consulting-framework-enqun-wang/]，数据科学咨询项目可大致分为五步：

    1. 需求定义与梳理（Requirement）
    2. 方案设计与讨论（Solution）
    3. 数据收集与分析（Data）
    4. 算法建模与调优（Algorithm）
    5. 结果展示与部署（Presentation/Launch）

    以上框架的前三项应在前期项目内完成，本文着重展示数据分析、算法的选择与调优，以及模型结果的呈现。
    """

    toc = """\
    ## 目录

    [一、数据导入](#section1)

    [二、探索性数据分析](#section2)

    [三、特征工程](#section3)

    [四、建模基准线及算法选择](#section4)

    [五、算法超参数调优](#section5)

    [六、算法建模](#section6)

    [七、模型表现评价](#section7)

    [八、模型解释](#section8)

    [九、总结](#section9)
    """

    section1 = """\
    <a id='section1'></a>
    ## 一、数据导入
    本项目建议将数据预处理工作以**管道（pipeline）**的形式封装成为算法库(Python package)，不同的数据预处理任务由相应的管道实现，最后进行拼接。
    """

    code1 = """\
    import pandas as pd
    pd.set_option('display.max_columns', 300)
    """

    section2 = """\
    <a id='section2'></a>
    ## 二、探索性数据分析
    **探索性数据分析（Exploratory Data Analysis，EDA）**，是指对已有的数据(特别是调查或观察得来的原始数据)在尽量少的先验假定下进行探索，通过作图、制表、方程拟合、计算特征量等手段探索数据的结构和规律的一种数据分析方法。特别是当我们对这些数据中的信息没有足够的经验，不知道该用何种传统统计方法进行分析时，探索性数据分析就会非常有效。探索性数据分析在二十世纪六十年代被提出，其方法由美国著名统计学家约翰•图基（John Tukey）命名。
    """

    code2 = """\
    import pandas_profiling as pp

    pp.ProfileReport()
    """

    section3 = """\
    <a id='section3'></a>
    ## 三、特征工程
    **特征工程（Feature Engineering）**，是指借助领域专业的经验深度挖掘数据的信息、构建特征变量，以提升模型的训练效果。数据和特征决定了机器学习的上限，而模型和算法是在逼近这个上限而已。由此可见，好的数据和特征是模型和算法发挥更大的作用的前提。

    **注意：**特征工程的工作不是一蹴而就的，往往伴随着多次的迭代，贯穿数据科学项目的始终。
    """

    code3 = """\
    """

    section4 = """\
    <a id='section4'></a>
    ## 四、建模基准线及算法选择


    """

    code4 = """\
    """

    section5 = """\
    <a id='section5'></a>
    ## 五、算法超参数调优

    机器学习相比于传统统计模型的进步其中一点体现在：通过算法调整模型的参数，这一过程的实质是最优化（Optimization）。而机器学习算法自身也引入了参数，区别于模型的参数，称为**超参数（Hyper Parameters）**。超参数的调优需要数据科学家结合自身的经验、数据的实际情况、模型表现要求，进行调优。
    """

    code5 = """\
    """

    section6 = """\
    <a id='section6'></a>
    ## 六、算法建模

    根据超参数调优结果，构建最终模型。
    """

    code6 = """\
    """

    section7 = """\
    <a id='section7'></a>
    ## 七、模型表现评价

    模型表现的评价通常会从两个方面进行：

    1. 数理层面：运用常见的模型评价指标；
    2. 业务层面：融入业务含义对模型进行评价。

    这这一步，本项目仅从数理层面对模型在测试集上的表现进行评价。**注意：**在数据建模开始前，我们已将数据划分为**训练集**与**测试集**，并且测试集自始至终没有也不应参与到模型训练中。
    """

    code7 = """\
    """

    section8 = """\
    <a id='section8'></a>
    ## 八、模型解释
    机器学习的可解释性（**Interpretability**）较低（即**黑盒子问题**）是困扰其大规模工业场景应用的重要原因，特别是在医学、银行等设涉及**人身安全、财产安全**的领域。在学术界，机器学习模型可解释性也是研究热点。

    目前，对于模型可解释性的研究主要集中在两个视角：

    1. 模型整体的解释：从模型整体得到特征重要性排序；
    2. 样本个体预测结果的解释：对于某个具体样本的预测结果（如分类结果）进行解释，即为什么一个样本会被模型预测为当前的结果，哪些特征起到了重要作用。

    针对模型整体的解释，`XGBoost`提供了三个特征重要性衡量的指标：

    - **weight** is the number of times a feature appears in a tree;
    - **gain** is the average gain of splits which use the feature;
    - **cover** is the average coverage of splits which use the feature where coverage is defined as the number of samples affected by the split.

    针对个体样本的解释，目前常用的方法如下：

    - `LIME`
    - `SHAP`
    """

    code8 = """\
    """

    section9 = """\
    <a id='section9'></a>
    ## 九、总结

    """

    nb['cell'] = [nbf.v4.new_markdown_cell(title),
                  nbf.v4.new_markdown_cell(toc),
                  nbf.v4.new_markdown_cell(section1),
                  nbf.v4.new_code_cell(code1),
                  nbf.v4.new_markdown_cell(section2),
                  nbf.v4.new_code_cell(code2),
                  nbf.v4.new_markdown_cell(section3),
                  nbf.v4.new_code_cell(code3),
                  nbf.v4.new_markdown_cell(section4),
                  nbf.v4.new_code_cell(code4),
                  nbf.v4.new_markdown_cell(section5),
                  nbf.v4.new_code_cell(code5),
                  nbf.v4.new_markdown_cell(section6),
                  nbf.v4.new_code_cell(code6),
                  nbf.v4.new_markdown_cell(section7),
                  nbf.v4.new_code_cell(code7),
                  nbf.v4.new_markdown_cell(section8),
                  nbf.v4.new_code_cell(code8),
                  nbf.v4.new_markdown_cell(section9)]

    fname = 'presentation.ipynb'

    with open(fname, 'w') as f:
        nbf.write(nb, f)

    return None

def main():
    setup()

if __name__ == '__main__':
    main()