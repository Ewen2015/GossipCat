#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""

import numpy as np
import matplotlib.pyplot as plt

class Explain(object):
    """Explain tree-based models with dtreeviz and SHAP."""
    def __init__(self, model, X, y, target, features, regression=False):
        """
        Args:
            model: A tree-based model, like XGBoost.
            X (np.narray): X values.
            y (np.array): y values.
            target (str): The target name.
            features (list): The list of features.
            regression (bool): Whether a regression model, defualt False. 
        """
        super(Explain, self).__init__()
        self.model = model
        self.X = X
        self.y = y
        self.target = target
        self.features = features
        self.regression = regression
    
    def tree(self, tree_index=0, class_names=None, show_node_labels=True, title="Decision Tree", orientation='TD', scale=1.5):
        """Plot the tree with dtreeviz.

        Args: 
            tree_index (int): The tree index of the model, default 0.
            class_names (list): [For classifiers] A dictionary or list of strings mapping class value to class name.
            show_node_labels (bool): Add "Node id" to top of each node in graph for educational purposes.
            title (str): The plot title.
            orientation (str): Is the tree top down, "TD", or left to right, "LR"?
            scale (float): Scale the width, height of the overall SVG preserving aspect ratio, default 1.5.

        Return:
            viz (dtreeviz): A dtreeviz instance.
        """
        import matplotlib.font_manager
        
        try:
            from dtreeviz.trees import dtreeviz
        except Exception as e:
            raise e
        
        if self.regression:
            class_names = None
        else:
            if class_names is None:
                class_names = np.unique(self.y).tolist()
        
        self.viz = dtreeviz(tree_model=self.model, 
                           x_data=self.X,
                           y_data=self.y,
                           target_name=self.target,
                           feature_names=self.features, 
                           class_names=class_names, 
                           tree_index=tree_index,
                           show_node_labels=show_node_labels,
                           orientation=orientation,
                           title=title,
                           scale=scale)
        return self.viz
    
    def feature_importance(self):
        """Plot the feature and SHAP variable importance with SHAP.
        """
        import shap
        
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer.shap_values(self.X)
        
        plot_type='bar'
        title='Feature Importance (by SHAP)'
        shap.summary_plot(shap_values=self.shap_values,
                          features=self.X,
                          feature_names=self.features, 
                          plot_type=plot_type,
                          show=False)
        plt.title(title)
        plt.show()
        
        plot_type='violin'
        title='The SHAP Variable Importance (by SHAP)'
        shap.summary_plot(shap_values=self.shap_values,
                          features=self.X,
                          feature_names=self.features, 
                          plot_type=plot_type,
                          show=False)
        plt.title(title)
        plt.gcf().axes[-1].set_aspect(100)
        plt.gcf().axes[-1].set_box_aspect(100)
        plt.show()
        return None