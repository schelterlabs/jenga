# Jenga

## Overview

__Jenga__ is an open source experimentation library that allows data science practititioners and researchers to study the effect of common data corruptions (e.g., missing values, broken character encodings) on the prediction quality of their ML models.

We design Jenga around three core abstractions: 

 * [Tasks](https://github.com/schelterlabs/jenga/tree/master/jenga/tasks) contain a raw dataset, an ML model and a prediction task
 * [Data corruptions](https://github.com/schelterlabs/jenga/tree/master/jenga/corruptions) take raw input data and randomly apply certain data errors to them (e.g., missing values)
 * [Evaluators](https://github.com/schelterlabs/jenga/tree/master/jenga/evaluation) take a task and data corruptions, and execute the evaluation by repeatedly corrupting the test data of the task, and recording the predictive performance of the model on the corrupted test data.

Jenga's goal is assist data scientists with detecting such errors early, so that they can protected their models against them. We provide a [jupyter notebook outlining the most basic usage of Jenga](basic-example.ipynb).

Note that you can implement custom tasks and data corruptions by extending the corresponding provided [base classes](https://github.com/schelterlabs/jenga/blob/master/jenga/basis.py).

We additionally provide three advanced usage examples of Jenga:
 * [Studying the impact of missing values](example-missing-value-imputation.ipynb)
 * [Stress testing a feature schema](example-schema-stresstest.ipynb)
 * [Evaluating the helpfulness of data augmentation for an image recognition task](example-image-augmentation.ipynb)

## Installation

Jenga requires Python 3.6 and virtualenv. You can get the Jenga code running as follows:

1. Checkout this git repository
1. Create a virtual environment with `python3.6 -m venv env`
1. Activate the environment with `source env/bin/activate`
1. Install the latest version of pip with `pip install --upgrade pip`
1. Install the dependencies with `pip install -r requirements.txt`

## Research

__Jenga__ is based on experiences and code from our ongoing research efforts:

 * Sebastian Schelter, Tammo Rukat, Felix Biessmann (2020). [Learning to Validate the Predictions of Black Box Classifiers on Unseen Data.](https://ssc.io/pdf/mod0077s.pdf) ACM SIGMOD. 
 * Tammo Rukat, Dustin Lange, Sebastian Schelter, Felix Biessmann (2020): [Towards Automated ML Model Monitoring: Measure, Improve and Quantify Data Quality.](https://ssc.io/pdf/autoops.pdf) ML Ops workshop at the Conference on Machine Learning and Systems&nbsp;(MLSys). 
 * Felix Biessmann, Tammo Rukat, Philipp Schmidt, Prathik Naidu, Sebastian Schelter, Andrey Taptunov, Dustin Lange, David Salinas (2019). [DataWig - Missing Value Imputation for Tables.](https://ssc.io/pdf/datawig.pdf) JMLR (open source track)
