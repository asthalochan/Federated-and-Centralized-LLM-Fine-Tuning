# Federated and Centralized LLM Fine-Tuning

This repository contains notebook-based experiments for:

- centralized fine-tuning of an LLM on an instruction dataset,
- federated fine-tuning with Flower and LoRA,
- privacy analysis using differential privacy, perplexity-based checks, and membership inference style evaluation.

The project uses medical question-answering style data and compares centralized training against federated fine-tuning for utility and privacy.

## Theory and Background

For the full theoretical explanation behind this project, see the Medium article:

- [Federated Fine-Tuning in LLMs: Why the Future of AI Privacy Starts Here](https://medium.com/@mohantaastha/federated-fine-tuning-in-llms-why-the-future-of-ai-privacy-starts-here-a0de34f8c613)

This article complements the repository by explaining the motivation, privacy context, and broader reasoning behind federated fine-tuning for LLMs.

## Project Overview

The workflow in this repo is split across three Jupyter notebooks:

1. `Centralized_LLM_Fine-tuning.ipynb`
   Centralized fine-tuning baseline, model prompting, and evaluation.
2. `Federated_LLM_Fine-tuning (1).ipynb`
   Federated instruction tuning with Flower, client partitioning, server orchestration, differential privacy, and result visualization.
3. `Keeping-LLM-private (1).ipynb`
   Privacy-focused analysis including perplexity comparisons, membership inference style scoring, ROC plotting, and extraction demos.

## Main Ideas Covered

- Parameter-efficient fine-tuning with LoRA
- Federated training with Flower
- Differential privacy in federated optimization
- Centralized vs federated utility comparison
- Prompting and inference with fine-tuned models
- Privacy evaluation with perplexity and MIA-style analysis

## Repository Structure

```text
.
|-- Centralized_LLM_Fine-tuning.ipynb
|-- Federated_LLM_Fine-tuning (1).ipynb
|-- Keeping-LLM-private (1).ipynb
|-- README.md
|-- requirements.txt
|-- conf/
|   |-- centralized.yaml
|   |-- centralized_full.yaml
|   |-- federated.yaml
|   |-- federated_full.yaml
|   `-- mia.yaml
|-- utils/
|   |-- LLM.py
|   |-- mia.py
|   |-- utils.py
|   |-- raw_data.csv
|   |-- fl_demo.csv
|   |-- cen_demo.csv
|   |-- ex1.txt
|   `-- ex2.txt
|-- my_centralized_model/
|   |-- adapter_config.json
|   `-- adapter_model.bin
`-- my_fl_model/
    `-- peft_2/
        |-- adapter_config.json
        `-- adapter_model.bin
```

## What Each Folder Does

### `conf/`

Hydra configuration files for experiments.

- `centralized.yaml`: lightweight centralized fine-tuning setup
- `centralized_full.yaml`: larger centralized configuration
- `federated.yaml`: smaller federated setup using `EleutherAI/pythia-70m`
- `federated_full.yaml`: larger federated setup using `mistralai/Mistral-7B-v0.1`
- `mia.yaml`: privacy evaluation configuration

### `utils/`

Core helper code used by the notebooks.

- `utils.py`
  Training utilities, Hydra config loading, dataset formatting, Flower client/server logic, LoRA model setup, visualization, and benchmark evaluation.
- `LLM.py`
  Wrapper classes for calling pretrained, centralized, and federated model endpoints through Fireworks.
- `mia.py`
  Privacy analysis utilities including perplexity scoring, model loading, MIA-style evaluation, and ROC analysis.

### Model Output Folders

- `my_centralized_model/`
  Stores LoRA adapter weights from centralized fine-tuning.
- `my_fl_model/`
  Stores LoRA adapter checkpoints produced during federated rounds.

## Datasets and Models

The configs in this repository reference:

- Dataset: `medalpaca/medical_meadow_medical_flashcards`
- Benchmark dataset: `bigbio/pubmed_qa`
- Small model: `EleutherAI/pythia-70m`
- Large model: `mistralai/Mistral-7B-v0.1`

## Requirements

Main dependencies:

- `flwr`
- `flwr_datasets`
- `ray`
- `hydra-core`
- `trl`
- `transformers`
- `peft`
- `bitsandbytes`
- `scikit-learn`
- `gradio`
- `fireworks-ai`

Install them with:

```bash
pip install -r requirements.txt
```

## Environment Setup

Some notebook cells use Fireworks-hosted models through `utils/LLM.py`, which expects:

```bash
FIREWORKS_API_KEY=your_api_key_here
```

You can place this in a `.env` file in the project root.

## How to Use

### 1. Centralized Fine-Tuning

Open `Centralized_LLM_Fine-tuning.ipynb` to:

- load the dataset,
- test a pretrained model,
- fine-tune a centralized baseline,
- compare pretrained and fine-tuned performance,
- run benchmark-style evaluation.

### 2. Federated Fine-Tuning

Open `Federated_LLM_Fine-tuning (1).ipynb` to:

- load federated config,
- partition the dataset across clients,
- create Flower clients and server strategy,
- apply differential privacy,
- simulate federated fine-tuning,
- save LoRA checkpoints,
- compare results with centralized training,
- estimate communication costs.

### 3. Privacy Analysis

Open `Keeping-LLM-private (1).ipynb` to:

- query centralized and federated models,
- measure perplexity changes,
- compare pretrained vs fine-tuned privacy behavior,
- inspect membership inference style metrics,
- visualize ROC curves,
- explore simple extraction examples.

## Notes on Training

- The smaller configs are more suitable for limited hardware.
- The larger configs expect a capable GPU environment.
- `bitsandbytes` quantization is used when CUDA is available.
- LoRA is used to reduce the number of trainable parameters during tuning.

## Example Config Highlights

The federated setup includes:

- configurable number of clients and rounds,
- client sampling via `fraction_fit`,
- client CPU/GPU resource settings,
- DP controls such as `noise_mult` and `clip_norm`,
- cosine-style learning rate scheduling across rounds.

## Output and Evaluation

The notebooks include utilities for:

- plotting validation accuracy comparisons,
- saving PEFT checkpoints,
- running PubMedQA-style evaluation,
- computing communication cost estimates,
- comparing privacy risk between centralized and federated fine-tuning.

## Suggested Cleanup Before GitHub Push

You may want to rename a few files for cleaner GitHub presentation:

- `Federated_LLM_Fine-tuning (1).ipynb`
- `Keeping-LLM-private (1).ipynb`
- `README (1).md` files inside saved model folders

These names usually appear when files were downloaded or duplicated locally.

## Future Improvements

- add notebook screenshots or sample outputs,
- include benchmark results in a table,
- export notebooks to scripts for reproducibility,
- add a license,
- add a `.gitignore` for model artifacts and notebook checkpoints.

## Summary

This repo demonstrates how to compare centralized and federated LLM fine-tuning under a privacy-aware setup. It combines Flower-based federated learning, LoRA fine-tuning, differential privacy, and privacy evaluation utilities in a notebook-friendly workflow.
