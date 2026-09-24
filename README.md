# LSTM Text Generator

## Overview

This project implements a character-level text generation system using an
LSTM (Long Short-Term Memory) neural network.

The model is trained on Shakespeare's text and learns to predict the next
character based on a sequence of previous characters.

## Task Requirements Covered

- Dataset loading
- Lowercase conversion
- Punctuation removal
- Character tokenization
- Input-output sequence creation
- Training/validation split
- Embedding layer
- LSTM layer
- Dense output layer with softmax
- Adam optimizer
- Sparse categorical crossentropy loss
- Early stopping
- Seed-based iterative text generation
- Sample generated outputs

## Technologies

- Python
- TensorFlow
- Keras
- NumPy
- LSTM
- Neural Networks

## Dataset

Shakespeare text dataset:

https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt

The dataset is downloaded automatically when `train.py` is run.

## Project Structure

```text
lstm-text-generator/
├── train.py
├── requirements.txt
├── README.md
├── generated_text.txt
├── .gitignore
└── lstm_text_generator.keras   # created after training
```

## Setup

### 1. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the project

```bash
python train.py
```

The first run downloads the dataset automatically.

## Model Architecture

```text
Input Character Sequence
        |
        v
Embedding Layer
        |
        v
LSTM Layer
        |
        v
Dense Layer
        |
        v
Softmax
        |
        v
Next Character
```

## Text Generation

The model is given a seed input such as:

```text
to be or not to be
```

It predicts the next character and repeatedly feeds the growing sequence
back into the model to generate new text.

Sample seed inputs used by the project:

- `to be or not to be`
- `the king`
- `love is`

Generated results are saved in `generated_text.txt`.

## Notes

Training time depends on the computer and TensorFlow environment. The
configuration in `train.py` is intentionally kept suitable for a practical
demonstration.
