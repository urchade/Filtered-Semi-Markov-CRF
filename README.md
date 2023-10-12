# Filtered Semi-Markov CRF

This repository provides a PyTorch implementation of the paper "Filtered Semi-Markov CRF," published in the Findings of EMNLP 2023. It focuses on implementing span-based Named Entity Recognition (NER) models that feature structured training and decoding.

## Table of Contents

- [Overview](#overview)
- [Training Algorithms](#training-algorithms)
- [Decoding Algorithms](#decoding-algorithms)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)

## Overview

The repository includes a variety of training and decoding algorithms to cater to different needs and use-cases for span-based NER.

## Training Algorithms

Here are the different training algorithms that have been implemented:

1. **Standard Span-Based NER with Local Objective**: This is the baseline algorithm for training the span-based NER model.
  
2. **Global Span Selection**: An implementation based on the model from Zaratiana et al., 2022.
   - 📝 [Read the paper](https://aclanthology.org/2022.umios-1.2/)

3. **Filtered Semi-Markov CRF**: This algorithm utilizes global span selection but adds label-dependent scoring and transition scores. It is essentially a filtered version of the original semi-CRF algorithm.
