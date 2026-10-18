"""
DLE602 Assessment 1
Twitter sentiment analysis with a bigram model.

How to run
1. python3 -m venv venv
2. source venv/bin/activate
3. pip install contractions emoji
4. Put the dictionaries in data/dictionaries/
5. Put the two csv files in data/datasets/
6. From the project folder, run: python main.py

The program uses 80% of the positive and negative tweets to train the bigram.
The other 20%, plus the neutral tweets, are only classified.
When the tables finish, you can type a tweet. Leave a line empty to stop.

Datasets
STSTd is the Stanford Twitter Sentiment Test set (Go, Bhayani and Huang, 2009):
https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
The file I used is testdata.manual.2009.06.14.csv

STSGd is the Stanford Twitter Sentiment Gold set (Saif, Fernandez, He and Alani, 2013).
Many of the original tweets no longer exist, so I used a public mirror that still
keeps the tweet text:
https://raw.githubusercontent.com/pollockj/world_mood/master/sts_gold_v03/sts_gold_tweet.csv

Zhao, J., Gui, X., and Zhang, X. (2018). Deep convolution neural networks
for Twitter sentiment analysis. IEEE Access, 6, 23253-23260.
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8244338
"""

from preprocessing import preprocessing_text
from bigram_model import train_datasets
from classify_tweets import classify_records, classify_tweet
from compare_results import compare_records


def run_datasets():
    datasets, models = train_datasets()

    for name, records in datasets.items():
        classify_records(records, models[name])
        comparison = compare_records(records)
        print(name, "test rows:", len(records))
        print(" original:", comparison["original"])
        print(" predicted:", comparison["predicted"])
        print(" same:", comparison["same"], "different:", comparison["different"])
        print(" accuracy:", round(comparison["accuracy"], 4))
        print("")

    return models


def client_paragraph_capture(model):
    print("What do you want to do?")
    twitterSentences = []

    while True:
        twitterSentence = input()
        if twitterSentence == "":
            break
        twitterSentences.append(twitterSentence)

    twitterParagraph = "\n".join(twitterSentences).strip()
    if twitterParagraph == "":
        return

    processed_text = preprocessing_text(twitterParagraph)
    print("-This is you text")
    print(processed_text)

    tokens = processed_text.split()
    predicted = classify_tweet(tokens, model)
    print("predicted:", predicted)


if __name__ == "__main__":
    models = run_datasets()
    client_paragraph_capture(models["STSTd"])