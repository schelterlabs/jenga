# Jenga

__Jenga__ is an experimentation library that allows data science practititioners and researchers to study the effect of common data corruptions (e.g., missing values, broken character encodings) on the prediction quality of their ML models.

We design Jenga around three core abstractions: 

 * [Tasks](https://github.com/schelterlabs/jenga/tree/master/jenga/tasks) contain a raw dataset, an ML model and a prediction task
 * [Data corruptions](https://github.com/schelterlabs/jenga/tree/master/jenga/corruptions) take raw input data and randomly apply certain data errors to them (e.g., missing values)
 * [Evaluators](https://github.com/schelterlabs/jenga/tree/master/jenga/evaluation) take a task and data corruptions, and execute the evaluation by repeatedly corrupting the test data of the task, and recording the predictive performance of the model on the corrupted test data.

Jenga's goal is assist data scientists with detecting such errors early, so that they can protected their models against them. We provide a jupyter notebook outlining the most basic usage of Jenga at 

 * [Predicting the helpfulness of video game reviews](example-reviews.ipynb)
 * [Distinguishing images of sneakers from images of ankle boots](example-shoes.ipynb)
 * [Predicting future movie ratings](example-movieratings.ipynb)

__Jenga__ is part of our ongoing research efforts:

 * Sebastian Schelter, Tammo Rukat, Felix Biessmann (2020). [Learning to Validate the Predictions of Black Box Classifiers on Unseen Data.](https://ssc.io/pdf/mod0077s.pdf) ACM SIGMOD. 
 * Tammo Rukat, Dustin Lange, Sebastian Schelter, Felix Biessmann (2020): [Towards Automated ML Model Monitoring: Measure, Improve and Quantify Data Quality.](https://ssc.io/pdf/autoops.pdf) ML Ops workshop at the Conference on Machine Learning and Systems&nbsp;(MLSys). 
