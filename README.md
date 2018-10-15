# Text Normalizer

[![License](https://img.shields.io/badge/license-GPL%203.0-green.svg)](https://opensource.org/licenses/GPL-3.0) 
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Django.svg) 
[![TensorFlow](https://img.shields.io/badge/-TensorFlow-orange.svg)](https://www.tensorflow.org/)

## Overview

<p align="center">
    <img src="https://i.imgur.com/j8sqVrE.png" alt="Project overview" />
</p>

## Dataset

The dataset that's going to be used for the experiments is the "Google Text Normalization Challenge" dataset available through the following URL: https://www.kaggle.com/google-nlu/text-normalization.

Example of a sentence contained in the dataset:

| Semiotic Class | Original Token | Normalized Token         |
| :------------- | :------------- | :----------------------- |
| PLAIN          | A              | \<self\>                 |
| PLAIN          | baby           | \<self\>                 |
| PLAIN          | giraffe        | \<self\>                 |
| PLAIN          | is             | \<self\>                 |
| MEASURE        | 6ft            | six feet                 |
| PLAIN          | tall           | \<self\>                 |
| PLAIN          | and            | \<self\>                 |
| PLAIN          | weighs         | \<self\>                 |
| MEASURE        | 150lb          | one hundred fifty pounds |
| PUNCT          | .              | sil                      |
| \<eos\>        | \<eos\>        |                          |

*Python implementation and more details to come*
