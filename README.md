## General purpose solution for data analysis and modeling

### Description

Currently this projects only handles tabular data in csv format and JSON files. The modeling approach is a binary classification task. 
The following pipeline uses a general approach to data analysis and modeling. The pipeline is divided into two parts:

1. Data analysis and feature engineering: reading data, performing EDA, basic feature engineeirng, data cleaning, and preprocessing data for modeling (missing values, one-hot encoding, etc.) EDA also provides useful plots to visualize the data and understand the problem better.

2. Modeling: Currently the projects uses tree-based models, the models area trained and evaluated using cross-validation. The models are tuned using optuna framewoek and the best model is selected based on the ROC AUC score. There are some useful plots to evalaute performace such as feature importance, ROC curve, and learning curves

Requirements:

1. Conda installed.


Steps:

1. Create a conda envirnoment conda env `conda env create -f environment.yml`

To see results:

Navigate to the `testing.ipynb` notebook and run the cells to perform the analysis and modeling.

### TODO

1. Add more models
2. Add more data preprocessing steps
3. Find a better usage for abstract classes
4. Remove hard-coded variables and refactor as parameters
5. Enhace the scope and by adding multiclass classification and regression tasks
6. Add more quality checks in the pre-commit-config file to improve code quality
7. Add more tests
8. Follow dessign principles. My attempt to follow as much as possible the OOP principles, but there is still room for improvement.
9. Add more documentation
10. Add type hints and improve docstrings