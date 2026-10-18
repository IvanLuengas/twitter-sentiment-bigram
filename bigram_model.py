from tokenize_tweets import prepare_datasets
import random


def count_bigrams(records):
    # unigram and bigram counts for one class
    unigram_counts = {}
    bigram_counts = {}
    token_total = 0

    for record in records:
        tokens = record["tokens"]
        token_total += len(tokens)
        for word in tokens:
            if word not in unigram_counts:
                unigram_counts[word] = 0
            unigram_counts[word] += 1

        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            if pair not in bigram_counts:
                bigram_counts[pair] = 0
            bigram_counts[pair] += 1

    return {
        "unigram_counts": unigram_counts,
        "bigram_counts": bigram_counts,
        "token_total": token_total,
    }


def train_bigram(records):
    # neutrals stay out of training; they are counted later
    positive_records = []
    negative_records = []

    for record in records:
        if record["label"] == "positive":
            positive_records.append(record)
        elif record["label"] == "negative":
            negative_records.append(record)

    model = {
        "positive": count_bigrams(positive_records),
        "negative": count_bigrams(negative_records),
    }
    return model


def split_records(records):
    # same csv rows: 80 percent train the bigram, 20 percent are only classified
    # neutrals never train
    positive_records = []
    negative_records = []
    neutral_records = []

    for record in records:
        if record["label"] == "positive":
            positive_records.append(record)
        elif record["label"] == "negative":
            negative_records.append(record)
        else:
            neutral_records.append(record)

    random.seed(1)
    random.shuffle(positive_records)
    random.shuffle(negative_records)

    positive_cut = int(len(positive_records) * 0.8)
    negative_cut = int(len(negative_records) * 0.8)

    train_records = positive_records[:positive_cut] + negative_records[:negative_cut]
    test_records = (
        positive_records[positive_cut:]
        + negative_records[negative_cut:]
        + neutral_records
    )
    return train_records, test_records


def train_datasets():
    datasets = prepare_datasets()
    models = {}
    test_datasets = {}

    for name, records in datasets.items():
        train_records, test_records = split_records(records)
        models[name] = train_bigram(train_records)
        test_datasets[name] = test_records

    return test_datasets, models


if __name__ == "__main__":
    datasets, models = train_datasets()
    for name in models:
        positive_model = models[name]["positive"]
        negative_model = models[name]["negative"]
        print(name)
        print(" positive tokens:", positive_model["token_total"])
        print(" positive bigrams:", len(positive_model["bigram_counts"]))
        print(" negative tokens:", negative_model["token_total"])
        print(" negative bigrams:", len(negative_model["bigram_counts"]))