# Jenga

## Overview

__Jenga__ is an open source experimentation library that allows data science practitioners and researchers to study
the effect of common data corruptions (e.g., missing values, broken character encodings) on the prediction quality of
their ML models.

We design Jenga around three core abstractions:

* [Tasks](tasks) contain a raw dataset, an ML model and a prediction task
* [Data corruptions](corruptions) take raw input data and randomly apply certain data errors to them (e.g., missing
  values)
* [Evaluators](evaluation) take a task and data corruptions, and execute the evaluation by repeatedly corrupting the
  test data of the task, and recording the predictive performance of the model on the corrupted test data.

Jenga's goal is assist data scientists with detecting such errors early, so that they can protected their models against
them. We provide a [jupyter notebook outlining the most basic usage of Jenga](notebooks/basic-example.ipynb).

Note that you can implement custom tasks and data corruptions by extending the corresponding
provided [base classes](https://github.com/schelterlabs/jenga/blob/master/jenga/basis.py).

We additionally provide three advanced usage examples of Jenga:

* [Studying the impact of missing values](notebooks/example-missing-value-imputation.ipynb)
* [Stress testing a feature schema](notebooks/example-schema-stresstest.ipynb)
* [Evaluating the helpfulness of data augmentation for an image recognition task](notebooks/example-image-augmentation.ipynb)

## Requirements

To proceed with the installation of Jenga, the following requirements must be met:

* Python between version 3.7 and 3.11 is required.
* An operating system different from Microsoft Windows is preferable; when using Windows, WSL must be used.

Both requirements are inherited from [TensorFlow](https://www.tensorflow.org/),
respectively the [`tensorflow-data-validation` package](https://github.com/tensorflow/data-validation),
which is required for Jenga's `validation` package extra.

## Installation

The following options are possible:

```bash
pip install jenga             # jenga is ready for the most corruptions (not images)
pip install jenga[all]        # install all dependencies, optimal for development
pip install jenga[image]      # also installs tensorflow ad image corruption/augmentation libraries
pip install jenga[validation] # also install tensorflow and tensorflow-data-validation necessary for SchemaStresstest
```

## Research

__Jenga__ is based on experiences and code from our ongoing research efforts:

* Sebastian Schelter, Tammo Rukat, Felix Biessmann (
  2020). [Learning to Validate the Predictions of Black Box Classifiers on Unseen Data.](https://ssc.io/pdf/mod0077s.pdf)
  ACM SIGMOD.
* Tammo Rukat, Dustin Lange, Sebastian Schelter, Felix Biessmann (
  2020): [Towards Automated ML Model Monitoring: Measure, Improve and Quantify Data Quality.](https://ssc.io/pdf/autoops.pdf)
  ML Ops workshop at the Conference on Machine Learning and Systems&nbsp;(MLSys).
* Felix Biessmann, Tammo Rukat, Philipp Schmidt, Prathik Naidu, Sebastian Schelter, Andrey Taptunov, Dustin Lange, David
  Salinas (2019). [DataWig - Missing Value Imputation for Tables.](https://ssc.io/pdf/datawig.pdf) JMLR (open source
  track)

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

## Installation for Development

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

Then take a look into the `notebooks` folder.

## Note

This project has been set up using PyScaffold 3.2.2 and the [dsproject extension] 0.4.
For details and usage information on PyScaffold see https://pyscaffold.org/.

[conda]: https://docs.conda.io/

[pre-commit]: https://pre-commit.com/

[Jupyter]: https://jupyter.org/

[nbstripout]: https://github.com/kynan/nbstripout

[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings

[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject
