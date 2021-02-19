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

In order to set up the necessary environment:

1. create an environment `jenga` with the help of [conda],
   ```
   conda env create -f environment.yaml
   ```
2. activate the new environment with
   ```
   conda activate jenga
   ```
3. install `jenga` with:
   ```
   python setup.py install # or `develop`
   ```

Optional and needed only once after `git clone`:

4. install several [pre-commit] git hooks with:
   ```
   pre-commit install
   ```
   and checkout the configuration under `.pre-commit-config.yaml`.
   The `-n, --no-verify` flag of `git commit` can be used to deactivate pre-commit hooks temporarily.

Then take a look into the `scripts` and `notebooks` folders.


## PyPI

(TODO: Not yet published)

The following options are possible:

```bash
pip install jenga             # jenga is ready for all types of corruptions
pip install jenga[all]        # install all dependencies, optimal for development
pip install jenga[image]      # also installs tensorflow necessary for ShoeCategorizationTask
pip install jenga[validation] # also install tensorflow-data-validation necessary for SchemaStresstest
```


## Dependency Management & Reproducibility

1. Always keep your abstract (unpinned) dependencies updated in `environment.yaml` and eventually
   in `setup.cfg` if you want to ship and install your package via `pip` later on.
2. Create concrete dependencies as `environment.lock.yaml` for the exact reproduction of your
   environment with:
   ```
   conda env export -n jenga -f environment.lock.yaml
   ```
   For multi-OS development, consider using `--no-builds` during the export.
3. Update your current environment with respect to a new `environment.lock.yaml` using:
   ```
   conda env update -f environment.lock.yaml --prune
   ```


## Research

__Jenga__ is based on experiences and code from our ongoing research efforts:

 * Sebastian Schelter, Tammo Rukat, Felix Biessmann (2020). [Learning to Validate the Predictions of Black Box Classifiers on Unseen Data.](https://ssc.io/pdf/mod0077s.pdf) ACM SIGMOD.
 * Tammo Rukat, Dustin Lange, Sebastian Schelter, Felix Biessmann (2020): [Towards Automated ML Model Monitoring: Measure, Improve and Quantify Data Quality.](https://ssc.io/pdf/autoops.pdf) ML Ops workshop at the Conference on Machine Learning and Systems&nbsp;(MLSys).
 * Felix Biessmann, Tammo Rukat, Philipp Schmidt, Prathik Naidu, Sebastian Schelter, Andrey Taptunov, Dustin Lange, David Salinas (2019). [DataWig - Missing Value Imputation for Tables.](https://ssc.io/pdf/datawig.pdf) JMLR (open source track)


## Note

This project has been set up using PyScaffold 3.2.2 and the [dsproject extension] 0.4.
For details and usage information on PyScaffold see https://pyscaffold.org/.

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject
