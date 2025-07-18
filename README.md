# Communication-Efficient Federated Learning for Fine Tuning LLM
This project explores the integration of Knowledge Distillation and Low-Rank Adaptation (LoRA) within a federated learning setup to fine-tune large language models with reduced communication cost.

## Environment & Prerequisites
- 4 × RTX 6000 GPUs
- Python 3.9+
- Install dependencies:
  ```bash
  pip install -r requirements.txt

## System Architecture
Below is the system design diagram, worked with during the model integration phase. Each client trains a student model with LoRA, and the server coordinates knowledge transfer from a pre-trained teacher.
<img width="854" height="636" alt="System Flow" src="https://github.com/user-attachments/assets/fad31587-237d-4eb9-9dfe-f9f4c3c2e4fb" />

## Experiment Workflow
### 1. Pre-train the Teacher Model
We first fine-tuned the teacher model on a public subset of the dataset to give it an initial performance edge.
```bash
PYTHONPATH=. python3 baselines/ft.py --teacher_ckpt bert-base-uncased --data_name cola --teacher_data_pct 20 --teacher_pretrain_epochs 5
```
In this case, the teacher model `bert-base-uncased` finetunes on the first 20% of the CoLA dataset for 5 epochs.

For SST2:
```bash
PYTHONPATH=. python3 baselines/ft.py --teacher_ckpt bert-base-uncased --data_name sst2 --teacher_data_pct 20 --teacher_pretrain_epochs 3
```
### 2. Run Our FL Scheme (FedKDLora)

For running 4 clients:
```bash
# view help
source multigpu/run-kd-lora.sh 4 -h
# run with the pretrained teacher model on the cola dataset. teacher_data_pct is used for the clients to figure out how much data they should use.
source multigpu/run-kd-lora.sh 4 --teacher_ckpt results/model/bert-base-uncased_cola_20pct_5epochs/ --data_name cola --teacher_data_pct 20
```
Results are stored in `results/*.json`.

### 3. Baselines for Comparison
```bash
# FedLoRA (FedAvg with LoRA)
source multigpu/run-lora.sh 4 --data_name cola
# FedAvg
source multigpu/run.sh 4 --data_name cola
```
## Datasets
We used datasets within the [`GLUE` benchmark](https://gluebenchmark.com/).
### [`SST2`](https://huggingface.co/datasets/glue/viewer/sst2)
[`SST2`](https://huggingface.co/datasets/glue/viewer/sst2) is a sentiment classification dataset. Each sample contains a sentence and the corresponding sentiment label. It contains 67k training samples and 1.6k testing samples.
### [`CoLA`](https://huggingface.co/datasets/glue/viewer/cola)
[`CoLA`](https://huggingface.co/datasets/glue/viewer/cola) is a test of grammatical correctness. Each sample contains a sentence and whether it is grammatically correct. It contains 9k training samples and 1k testing samples.

## Packages used
### Huggingface
We used [Huggingface](https://huggingface.co/) libraries, all of which are written in Python.
1. [`transformers`](https://huggingface.co/docs/transformers/) (1.2M LOC): For downloading pre-trained LLMs (BERT and DistilBERT in our case) into pytorch format. [Github](https://github.com/huggingface/transformers), 1.2 million LOC.
1. [`datasets`](https://huggingface.co/docs/datasets/) (87k LOC): For loading the `SST2` and `CoLA` datasets that we used for evaluation. [Github](https://github.com/huggingface/datasets/)
1. [`peft`](https://huggingface.co/docs/peft/) (112k LOC): For transforming large models into parameter-efficient versions (LoRA). [Github](https://github.com/huggingface/peft/)

### Flower
[Flower](https://flower.dev/) (141k LOC) is a simple FL framework written in Python, but also supports FL with mobile devices (iOS/Android). It provides a standard FedAvg implementation, and we implemented FedLoRA and our FL scheme (FedKDLoRA) ourselves. [Github](https://github.com/adap/flower)

### PyTorch
We use [PyTorch](https://pytorch.org/) (220k LOC) for model training. Written in C++ with Python bindings. [Github](https://github.com/pytorch/pytorch)

## Performance Metrics
We evaluate all the schemes (FedAvg, FedLora, FedKDLora) via accuracy and loss with respect to the test dataset at every communication round. We also time the total runtime of each scheme.


## Result
Individual Model Comparison
## Result
Individual Model Comparison  
<img width="854" alt="Individual Model Comparison" src="https://github.com/user-attachments/assets/7866f2ca-a7a5-4d96-9c63-07c0c798391a" />

Easy Case  
<img width="854" alt="Esay Case" src="https://github.com/user-attachments/assets/4a91d785-5eb6-492f-9e73-0dd06b44943d" />

Hard Case  
<img width="854" alt="Hard Case" src="https://github.com/user-attachments/assets/4ce8003c-17e3-4916-8a55-6ca7f520b9e2" />

## Key Observations
- FedKDLoRA performs better than FedLoRA on CoLA dataset in early rounds but flattens later.
- On SST2, FedKDLoRA consistently shows higher stability across different LoRA ranks.
- Runtime increases with LoRA rank, but marginal gains in accuracy diminish after R=16.

## Note
> This repository is part of a group project completed for Georgia Tech **CS6220: Big Data Systems & Analytics** (Fall 2023).  
> Shared code and core documents were collaboratively developed by the team.  
> This repository reflects my personal understanding and organization of the project, including selective highlights, documentation, and key observations.
