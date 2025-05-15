# ML Project 1 : Heart Diseases Detection Using Machine Learning Techniques

This repository presents our machine learning project focused on detecting heart disease based on personal lifestyle factors. We applied seven different algorithms, including (Ridge) Logistic Regression, Mean Squared Error Gradient Descent (Stochastic), Least Squares, Ridge Regression, and Support Vector Machines (SVM).



**Authors:** Mahmoud Dokmak, Adam Ben Slama, Jianan Xu

<hr style="clear:both">

## Overview

Heart disease remains one of the leading causes of
death globally, prompting an urgent need for more accessible
and cost-effective diagnostic methods than conventional medical consultations. Leveraging the advancements in machine
learning and the increasing computational power of modern
computers, we can now explore and apply various algorithms to
enhance the detection of heart conditions. This project aims to find the most efficient basic machine learning algorithm for predicting heart disease.



## Data
To accomplish this task, we utilized data from the Behavioral Risk Factor Surveillance System (BRFSS), which collects extensive lifestyle and medical information, and their heart disease status

⭐ In order to run `run.ipynb`, please put csv files (datasets) under `./data/dataset`.

## Description

The repository has the following structure:
- `run.ipynb`: A Jupyter Notebook that documents every step of our project. Running this notebook from start to finish will takes much time (about an hour) so we commented the cross validation part and used directly our best parameters so it can run faster (less than 5 minutes) and still produce our best submission score on AICrowd.
- `implementations.py`: This file contains 6 of the 7 regressions and classifications algorithms that we tested.
- `svm.py`: This file contains the Support Vector Machine algorithm, that we put in a separate file because it is not really a basic algorithm.
- `helpers.py`: Functions for data loading and creating the submission file.
- `myutils.py`: Contains useful functions needed across various libraries.
- `cross_validation.py`: Functions for selecting the best hyperparameters of the model using 5-fold cross-validation and ploting interesting graphs.
- `data_processing.py`: Functions for data processing and data analysis.


## Libraries
List of libraries used:
- `numpy`
- `matplotlib`

The entire project is implemented using these two basic libraries. `numpy` is used for the computations and matplotlib is used for graph visualisation.
