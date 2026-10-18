# Twitter Sentiment Analysis with Bigram Model

DLE602 Assessment 1 — Sentiment classification of tweets using a simple bigram language model.

This project implements a basic natural language processing pipeline to classify tweets as **positive** or **negative**. It uses a bigram model trained on a portion of the data and evaluates performance on held-out tweets (including neutral ones).

## Features

- Text preprocessing pipeline (contractions, emojis/emoticons, slang expansion, stopword removal, URL and user mention handling)
- Bigram model training on positive and negative tweets
- Classification of tweets based on bigram probabilities
- Evaluation against labeled datasets
- Interactive mode to classify new tweets from the command line

## Technologies

- **Python 3**
- `contractions` — expand English contractions
- `emoji` — convert emojis to text
- Standard library: `csv`, `json`, `re`, `pathlib`, `random`

## Datasets

Two classic Twitter sentiment datasets are used:

| Dataset | Description | Source |
|---------|-------------|--------|
| **STSTd** | Stanford Twitter Sentiment Test set (manual annotations) | Go, Bhayani & Huang (2009). File: `testdata.manual.2009.06.14.csv` |
| **STSGd** | Stanford Twitter Sentiment Gold set | Saif, Fernandez, He & Alani (2013). Public mirror used because many original tweets are no longer available. |

> **Note**: The original datasets are from 2009–2013. Many tweets no longer exist on Twitter/X. The project expects the CSV files to be placed in `data/datasets/`.

## Project Structure
├── main.py                 # Entry point — trains, evaluates, and runs interactive mode
├── preprocessing.py        # Text cleaning pipeline
├── tokenize_tweets.py      # Tokenization and dataset preparation
├── dataset_loader.py       # Load and count labels from CSVs
├── bigram_model.py         # Train bigram counts (positive / negative)
├── classify_tweets.py      # Classify records and single tweets
├── compare_results.py      # Compare predictions vs ground truth
├── data/
│   ├── datasets/           # Place the two CSV files here
│   └── dictionaries/       # slang_dict.json, emoticon_dict.json, van_stoplist.txt
└── requirements.txt


## Installation

```bash
# Clone the repository
git clone https://github.com/IvanLuengas/twitter-sentiment-bigram.git
cd twitter-sentiment-bigram

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt