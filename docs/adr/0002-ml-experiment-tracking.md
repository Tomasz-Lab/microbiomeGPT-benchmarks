# 0002. ML experiment tracking

Status: Accepted
Date: 2026-11-09

## Context
In our project, we plan a lot of tasks like benchmarking different model architectures, etc. So there is a need, after benchmarking, to easily review and compare metrics from different models in order to pick the best one for a given scenario.

Currently, model results are tracked inside CSV files or in Hydra configuration YAML files, but there is no tool or ready solution to easily read these files/results and make comparisons to easily spot the best model in any context.

On the market, there are two main tools:
- MLflow
- Weights & Biases

that seem to resolve this problem.

## Decision
1. We decided to pick Weights & Biases for a couple of reasons:
    - it's hosted by them, so there is no need to host this software on our side
    - compared to MLflow, it offers richer, interactive visualizations out of the box:
        - grouping and filtering runs (e.g. by model architecture or dataset) and comparing them on shared plots
        - parallel coordinates plots and parameter importance panels, which help to see which hyperparameters affect the results
        - custom, configurable dashboards per project
    - it automatically logs system metrics (GPU/CPU usage, memory), which helps compare the resource cost of different models
    - it also has the ability to build reports directly on the web and share them with the team
    - it allows easy integration with the PyTorch Lightning module
    - it also has an offline mode, so running code on a cluster without internet is possible
    - as Sano is an organization, we have access to a free Weights & Biases tier, because normally it is paid software
    - we have around 200 GB of storage for free for our artifacts (like model weights, datasets, etc.)
2. We decided to create an organization `sano-centre-for-computational-medicine-microbiome-gpt` where the results of our work will be stored.


## Consequences
Positive:
- we have a ready-to-use framework for ML experiment tracking
- it's ready to use very fast: just create an account, get an API key, integrate with Lightning and you have access
- comparing results of different models becomes much easier thanks to grouping runs, shared plots and dashboards, instead of manually reading CSV/YAML files

Negative:
- it's fully hosted by W&B, so if something happens to Weights & Biases, we may lose our experiment history
- our logging code becomes tied to the W&B API (vendor lock-in), so migrating to another tool will require some work
- we are working with sensitive or non-public data, so not everything can be sent to an external service

## Follow-up
- Decide what can be logged to W&B and what must stay local
- Decide how we keep a copy of our experiment history and where we migrate if W&B is no longer available


