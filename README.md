# NLP Meta-GPT Style Transfer

Welcome to the NLP Meta-GPT Style Transfer project! This project leverages the Model-Agnostic Meta-Learning (MAML) approach to train a GPT-2 model, enabling it to perform few-shot learning for any style transfer task, including styles not seen during the training phase. Utilizing the StylePTB dataset, our model showcases the flexibility and power of adapting to new styles with minimal examples.

## Overview

The goal of this project is to advance the capabilities of language models in understanding and generating text across a diverse range of writing styles. By integrating MAML with GPT-2, we enable our model to quickly adapt to new styles with only a few examples.

## Features

- **Few-shot Learning**: Our model can learn to transfer styles with just a few examples, making it highly efficient and adaptable.
- **Versatility**: Capable of handling a wide range of styles, even those not encountered during training.
- **Easy Integration**: Designed to be easily integrated into existing NLP pipelines and applications.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.7 or higher
- Transformers library

### Installation

Clone the repository to your local machine:

### Download Project

```bash
git clone https://github.com/ArthasL1/NLP-Meta-GPT-Style-Transfer.git
cd NLP-Meta-GPT-Style-Transfer
```

## Split Data
We provide two ways to split data, one is to divide it according to the specified number, and the other is to divide it according to the proportion. There are two files in total: **Split_data_by_numbers.py** and **Split_data_by_proportion.py**
### Split Data By Numbers
In **Split_data_by_numbers.py** it accepts 4 parameters, namely numbers of training samples, numbers of validation samples, numbers of testing samples and the name of the dataset to be created.

```python
python Split_data_by_numbers.py 2400 200 70 "Dataset_1"
```

### Split Data By Proportions
In **Split_data_by_proportion.py** it accepts 5 parameters, namely numbers of training samples, numbers of validation samples, numbers of testing samples in meta-training, numbers of testing samples in meta-testing, and the name of the dataset to be created.

```python
python Split_data_by_proportion.py 0.7 0.1 0.1 0.9 "Dataset_2"
```
