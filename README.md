# Twitter-NER-BiLSTM-CRF
This project tackles the challenge of recognizing named entities (PER, LOC, ORG, etc.) in noisy, user-generated Twitter data. It compares a CRF-based tagger with a deep BiLSTM-CRF model.

## Project Overiew 

The goal is to detect entities in tweets using two supervised sequence labeling techniques:
1. Conditional Random Fields (CRF)
2. BiLSTM combined with CRF (BiLSTM-CRF)
The BiLSTM-CRF model learns contextual word representations and jointly decodes tag sequences using the Viterbi algorithm.

## Data
1. Source: Broad Twitter Corpus (pre-tokenized with BIO entity labels)
2. Format: CoNLL-style with word, POS, chunk, and NER tag per line
3. Entity tags:
- PER (Person), ORG (Organization), LOC (Location), MISC

## Models Trained

| Model      | Description                                           |
| ---------- | ----------------------------------------------------- |
| CRF        | Uses transition features only                         |
| BiLSTM-CRF | Learns contextual word embeddings and decodes via CRF |

## Libraries Used
```
torch
scikit-learn
nltk
pandas / numpy
```

Install with:
```
pip install -r requirements.txt
```

## Methodlogy 
- BIO tagging used for sequence labeling
- CRF:
  1. Features: previous/next words, word shape, capitalization
  2. Features: previous/next words, word shape, capitalization
- BiLSTM-CRF:
  1. Word embeddings + BiLSTM layers
  2. CRF on top for joint decoding
  3. Viterbi decoding for final tag prediction

## Training Details
| Parameter       | Value |
| --------------- | ----- |
| Embedding Dim   | 100   |
| LSTM Hidden Dim | 128   |
| Optimizer       | Adam  |
| Learning Rate   | 0.001 |
| Epochs          | 20    |
| Batch Size      | 32    |

## Results 

<img width="312" alt="Screenshot 2025-06-30 at 1 10 41 AM" src="https://github.com/user-attachments/assets/289f2e29-724c-4418-b7e9-621674170f03" />

| Metric           | CRF    | BiLSTM-CRF |
| ---------------- | ------ | ---------- |
| Accuracy         | 0.8907 | 0.9153     |
| F1 Score (Macro) | 0.4587 | 0.5617     |



## Future Enhancements
1. Add character-level embeddings for better handling of misspellings
2. Replace word embeddings with contextual transformers (e.g., BERTweet)


