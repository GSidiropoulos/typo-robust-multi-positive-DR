# Multi-Positive Contrastive Learning for Robust DR
This repository contains the code for the paper [Improving the Robustness of Dense Retrievers Against Typos via Multi-Positive Contrastive Learning](https://link.springer.com/chapter/10.1007/978-3-031-56063-7_21), ECIR 2024.

##
The code is built on top of the code released for the paper [Typo-Robust Representation Learning for Dense Retrieval](https://github.com/panuthept/DST-DenseRetrieval/tree/main?tab=readme-ov-file)

## Installation
Use ```setup.sh``` script to install this repository and Python dependency packages:
```
sh setup.sh
```

## Download Model Checkpoints
You can download a pre-trained model checkpoint for our model [here](https://surfdrive.surf.nl/files/index.php/s/7aaQV6Pn63VYfYY).

## Train Model
To train the retriever model, run the following command:
```
sbatch train.job
```

## Encode Corpus and Queries and Retrieve
To encode the whole corpus and the queries and then retrieve, run the following command:
```
sbatch retrieve.job
```

## Evaluation
To evaluate the model using the following evaluation script, you need to install [trec_eval](https://github.com/usnistgov/trec_eval).
```
git clone https://github.com/usnistgov/trec_eval.git
cd trec_eval
make
```

Evaluate the retrieval results, using the following command:
```
sbatch eval.job
```
