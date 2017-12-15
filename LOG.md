# Log

#### Dec 14, 2017
---
**GossipCat 0.1.24**

- Split `Report` from `core` as a class.
- Polished `SimAnneal`.

#### Dec 13, 2017
---
**GossipCat 0.1.10**

Tested and modified functions. Add `param_1` as default params for `simAnneal` function.

#### Dec 12, 2017
---
**gossipcat.py**

Add several new functions to do feature engineering:
- `features_dup`: This function checks first n_head rows and obtains duplicated features.
- `features_clf`: This function divides features into sublists according to their data type.
- `corr_pairs`: This function computes correlated feature pairs with correlated coefficient larger than gamma.
- `features_new`: This function builds new features based on correlated feature pairs if the new feature has an AUC greater than auc_score with logistic regression. 

#### Dec 11, 2017
---
**gossipcat.py**

- `glimpse`: This function prints a general infomation of the dataset and plot the distribution of target value. 
- `simAnneal`: This function uses the simulated annealing to find the optimal hyper parameters and return an optimized classifier.
- `report`: This function prints model report with a classifier on test dataset.
- `report_CM`: This function prints the recall rate of the classifier on test data and plots out confusion matrix.
- `report_PR`: This function plots precision-recall curve and gives average precision.





