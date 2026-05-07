# Sequential vs Frequency Approach on Audio Classification

**Course:** Kecerdasan Komputasional — Institut Teknologi Sepuluh Nopember

## Overview

This project investigates two fundamentally different approaches to audio sound classification and evaluates which paradigm performs better under varying real world conditions.

The two approaches compared:

- **Sequential modeling** using Hidden Markov Models (HMM), which captures temporal structure in audio signals
- **Frequency based modeling** using Multinomial Naive Bayes, which treats audio as a distribution of spectral components

## Dataset

Urban sound classification dataset containing audio recordings across multiple sound categories. Metadata is provided in `metadata_wavsonly.csv`.

## Methodology

### Feature Extraction
Audio signals are converted to **MFCC (Mel Frequency Cepstral Coefficients)** features, which compress the spectral envelope of a sound into a compact representation widely used in speech and audio processing.

### Vector Quantization
Raw MFCC features are discretized into a codebook using **K Means clustering**, converting continuous feature vectors into symbolic sequences suitable for both HMM and Naive Bayes.

### Models

**Hidden Markov Model (HMM)**
- Treats each audio clip as a sequence of observations
- Models the probability of transitioning between hidden states over time
- Suited for sounds with temporal dynamics

**Multinomial Naive Bayes**
- Treats each audio clip as a bag of codeword frequencies
- Ignores temporal ordering, focuses purely on occurrence statistics
- Fast inference, strong baseline for frequency dominant sounds

### Experimental Scenarios

Three evaluation axes:

| Scenario | Variable |
|---|---|
| Codebook size sensitivity | Varied K from 32 to 512 |
| Training data size impact | Varied training set from 20% to 100% |
| Audio duration robustness | Tested on short vs long clips |

## Results

- **Naive Bayes** achieved up to **90% accuracy** and **95% precision** at full training data
- **HMM** showed competitive performance at smaller codebook sizes (85% accuracy at 32 clusters)
- Naive Bayes consistently outperformed HMM in frequency dominant classification tasks
- HMM retained value in smaller codebook regimes where temporal modeling helps

## Key Finding

For urban sound classification where temporal structure is not the primary discriminant, frequency based approaches (Naive Bayes) are more accurate, faster, and more data efficient than sequential models (HMM). However, HMM remains relevant when codebook granularity is limited.

## Files

| File | Description |
|---|---|
| `main.ipynb` | Full experiment notebook: preprocessing, training, evaluation |
| `metadata_wavsonly.csv` | Dataset metadata with file paths and labels |
| `dataset.zip` | Raw audio dataset |

## Tech Stack

Python, Librosa, scikit learn, NumPy, Matplotlib
