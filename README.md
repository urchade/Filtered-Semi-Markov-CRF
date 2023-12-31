# Filtered Semi-Markov CRF

This repository provides a PyTorch implementation of the paper "Filtered Semi-Markov CRF," published in the Findings of EMNLP 2023. It focuses on implementing span-based Named Entity Recognition (NER) models that feature structured training and decoding.

## Table of Contents

- [Overview](#overview)
- [Training Algorithms](#training-algorithms)
- [Decoding Algorithms](#decoding-algorithms)
- [Configuration Options](#configuration-options)


## Overview

The repository includes a variety of training and decoding algorithms to cater to different needs and use-cases for span-based NER.

## Training Algorithms

Here are the different training algorithms that have been implemented:

1. **Filtered Semi-Markov CRF**: This algorithm utilizes global span selection but adds label-dependent scoring and transition scores. It is essentially a filtered version of the original semi-CRF algorithm.
   - 📝 [Read the paper](https://arxiv.org/abs/2311.18028)
   
2. **Standard Span-Based NER with Local Objective**: This is the baseline algorithm for training the span-based NER model. Our paper contains details about this:
   - 📝 [Read the paper](https://aclanthology.org/2022.umios-1.1/)
     
3. **Global Span Selection**: An implementation based on the model from Zaratiana et al., 2022a.
   - 📝 [Read the paper](https://aclanthology.org/2022.umios-1.2/)

## Decoding Algorithms

The implemented decoding algorithms aim to return non-overlapping spans. The following algorithms are available:

1. **Greedy Decoding**: Returns the first best non-overlapping spans.
  
2. **Exact Decoding**: Returns spans with the highest sum of scores.
  
3. **Exhaustive Search**: Utilizes an arbitrary scoring function to return spans with the maximum score.
   - This has been proposed in our [Zaratiana et al., 2022b](https://aclanthology.org/2022.umios-1.1/)


## Configuration Options

To configure the model and decoding algorithm, modify the configuration file (see config/conll.yaml as an example) as described below:

### Model variants

#### [Named Entity Recognition as Structured Span Prediction](https://aclanthology.org/2022.umios-1.1) (Zaratiana et al., UM-IoS 2022a)
```plaintext
model_type: "standard"
decoding: "greedy" or "global" or "global_mean"
```

#### [Filtered Semi-Markov CRF](https://urchade.github.io/assets/Filtered_Semi_CRF.pdf) (Zaratiana et al., EMNLP 2023):
```plaintext
model_type: "fsemicrf"
decoding: "global"
```

#### [Global Span Selection for Named Entity Recognition](https://aclanthology.org/2022.umios-1.2) (Zaratiana et al., UM-IoS 2022b):
```plaintext
model_type: "gss"
decoding: "global"
```

### Alternatives
* Options for **decoding** parameter:
```plaintext
- 'global': maximize sum of span scores
- 'global_mean': maximize average of span scores
- 'greedy': greedy span selection
```

* Options for **model_type** parameter:
```plaintext
- 'standard': Standard Span-Based NER loss (span-level NLL)
- 'fsemicrf': Filtered Semi-Markov CRF loss 
- 'gss': Global Span Selection loss
```

## Citation

If you find this code useful in your research, please consider citing our papers


```bibtex
@misc{zaratiana2023filtered,
      title={Filtered Semi-Markov CRF}, 
      author={Urchade Zaratiana and Nadi Tomeh and Niama El Khbir and Pierre Holat and Thierry Charnois},
      year={2023},
      eprint={2311.18028},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

```bibtex
@inproceedings{zaratiana-etal-2022-global,
    title = "Global Span Selection for Named Entity Recognition",
    author = "Zaratiana, Urchade  and
      Elkhbir, Niama  and
      Holat, Pierre  and
      Tomeh, Nadi  and
      Charnois, Thierry",
    booktitle = "Proceedings of the Workshop on Unimodal and Multimodal Induction of Linguistic Structures (UM-IoS)",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.umios-1.2",
    doi = "10.18653/v1/2022.umios-1.2",
    pages = "11--17",
    abstract = "Named Entity Recognition (NER) is an important task in Natural Language Processing with applications in many domains. In this paper, we describe a novel approach to named entity recognition, in which we output a set of spans (i.e., segmentations) by maximizing a global score. During training, we optimize our model by maximizing the probability of the gold segmentation. During inference, we use dynamic programming to select the best segmentation under a linear time complexity. We prove that our approach outperforms CRF and semi-CRF models for Named Entity Recognition. We will make our code publicly available.",
}
```

```bibtex
@inproceedings{zaratiana-etal-2022-named,
    title = "Named Entity Recognition as Structured Span Prediction",
    author = "Zaratiana, Urchade  and
      Tomeh, Nadi  and
      Holat, Pierre  and
      Charnois, Thierry",
    booktitle = "Proceedings of the Workshop on Unimodal and Multimodal Induction of Linguistic Structures (UM-IoS)",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.umios-1.1",
    doi = "10.18653/v1/2022.umios-1.1",
    pages = "1--10",
    abstract = "Named Entity Recognition (NER) is an important task in Natural Language Processing with applications in many domains. While the dominant paradigm of NER is sequence labelling, span-based approaches have become very popular in recent times but are less well understood. In this work, we study different aspects of span-based NER, namely the span representation, learning strategy, and decoding algorithms to avoid span overlap. We also propose an exact algorithm that efficiently finds the set of non-overlapping spans that maximizes a global score, given a list of candidate spans. We performed our study on three benchmark NER datasets from different domains. We make our code publicly available at \url{https://github.com/urchade/span-structured-prediction}.",
}
```
