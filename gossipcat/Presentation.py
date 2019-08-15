#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""

import nbformat as nbf 
        

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
四、建模基准线及算法选择


"""

code4 = """\
"""

section15 = """\
<a id='section5'></a>
五、算法超参数调优

"""

code5 = """\
"""

section6 = """\
<a id='section6'></a>
六、算法建模

"""

code6 = """\
"""

section7 = """\
<a id='section7'></a>
七、模型表现评价

"""

code7 = """\
"""

section8 = """\
<a id='section8'></a>
八、模型解释

"""

code8 = """\
"""

section9 = """\
<a id='section9'></a>
九、总结

"""



def setup():
    nb = nbf.v4.new_notebook()



if __name__ == '__main__':
    main()
















