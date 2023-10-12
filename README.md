# Filtered Semi-Markov CRF

This repository provides a PyTorch implementation of the paper "Filtered Semi-Markov CRF," published in the Findings of EMNLP 2023. It focuses on implementing span-based Named Entity Recognition (NER) models that feature structured training and decoding.

## Table of Contents

- [Overview](#overview)
- [Training Algorithms](#training-algorithms)
- [Decoding Algorithms](#decoding-algorithms)


## Overview

The repository includes a variety of training and decoding algorithms to cater to different needs and use-cases for span-based NER.

## Training Algorithms

Here are the different training algorithms that have been implemented:

1. **Standard Span-Based NER with Local Objective**: This is the baseline algorithm for training the span-based NER model.
  
2. **Global Span Selection**: An implementation based on the model from Zaratiana et al., 2022a.
   - 📝 [Read the paper](https://aclanthology.org/2022.umios-1.2/)

3. **Filtered Semi-Markov CRF**: This algorithm utilizes global span selection but adds label-dependent scoring and transition scores. It is essentially a filtered version of the original semi-CRF algorithm.

## Decoding Algorithms

The implemented decoding algorithms aim to return non-overlapping spans. The following algorithms are available:

1. **Greedy Decoding**: Returns the first best non-overlapping spans.
  
2. **Exact Decoding**: Returns spans with the highest sum of scores.
  
3. **Exhaustive Search**: Utilizes an arbitrary scoring function to return spans with the maximum score.
   - This has been proposed in our [Zaratiana et al., 2022b](https://aclanthology.org/2022.umios-1.1/)


## Configuration Options

To configure the model and decoding algorithm, modify the configuration file as described below:

#### For Filtered Semi-Markov CRF:
```plaintext
model_type: "fsemicrf"
decoding: "global"
```

#### For Global Span Selection:
```plaintext
model_type: "gss"
decoding: "global"
```

#### For Standard Model:
```plaintext
model_type: "standard"
decoding: "greedy"
```

### List of Decoding Options
```plaintext
- 'global': maximize sum of span scores
- 'global_mean': maximize average of span scores
- 'greedy': greedy span selection
```

### List of model_type Options
```plaintext
- 'standard': Standard Span-Based NER
- 'fsemicrf': Filtered Semi-Markov CRF
- 'gss': Global Span Selection
```
