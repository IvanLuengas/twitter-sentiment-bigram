from bigram_model import train_datasets


def vocabulary_size(model):
    words = {}
    for word in model["positive"]["unigram_counts"]:
        words[word] = True
    for word in model["negative"]["unigram_counts"]:
        words[word] = True
    return len(words)


def word_probability(model_side, previous_word, word, vocab_size):
    # Laplace: a pair that was never seen does not become zero
    unigram_counts = model_side["unigram_counts"]
    bigram_counts = model_side["bigram_counts"]

    if previous_word is None:
        count = unigram_counts.get(word, 0)
        total = model_side["token_total"]
        return (count + 1) / (total + vocab_size)

    pair = (previous_word, word)
    pair_count = bigram_counts.get(pair, 0)
    previous_count = unigram_counts.get(previous_word, 0)
    return (pair_count + 1) / (previous_count + vocab_size)


negation_words = {"not", "no", "never"}
# placeholders from preprocessing, they are not sentiment words
ignored_words = {"user", "url"}


def label_words(tokens, model):
    # "not" does not vote. It flips the next word, so "not good" is one signal.
    vocab_size = vocabulary_size(model)
    labels = []
    previous_word = None
    flip_next = False

    for word in tokens:
        if word in negation_words:
            labels.append("skip")
            flip_next = True
            previous_word = word
            continue

        if word in ignored_words:
            labels.append("skip")
            previous_word = word
            continue

        positive_probability = word_probability(
            model["positive"], previous_word, word, vocab_size
        )
        negative_probability = word_probability(
            model["negative"], previous_word, word, vocab_size
        )

        if positive_probability > negative_probability:
            label = "positive"
        elif negative_probability > positive_probability:
            label = "negative"
        else:
            label = "tie"

        if flip_next and label == "positive":
            label = "negative"
        elif flip_next and label == "negative":
            label = "positive"
        flip_next = False

        labels.append(label)
        previous_word = word

    return labels


def classify_tweet(tokens, model):
    # one fourth rule: both sides over the line, or neither, is neutral
    if len(tokens) == 0:
        return "neutral"

    labels = label_words(tokens, model)
    positive_words = 0
    negative_words = 0
    voting_words = 0

    for label in labels:
        if label == "skip":
            continue
        voting_words += 1
        if label == "positive":
            positive_words += 1
        elif label == "negative":
            negative_words += 1

    if voting_words == 0:
        return "neutral"

    positive_share = positive_words / voting_words
    negative_share = negative_words / voting_words

    if positive_share >= 0.25 and negative_share >= 0.25:
        return "neutral"
    if positive_share >= 0.25:
        return "positive"
    if negative_share >= 0.25:
        return "negative"
    return "neutral"


def classify_records(records, model):
    for record in records:
        record["predicted"] = classify_tweet(record["tokens"], model)
    return records


def count_predicted(records):
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for record in records:
        counts[record["predicted"]] += 1
    return counts


if __name__ == "__main__":
    datasets, models = train_datasets()
    for name, records in datasets.items():
        classify_records(records, models[name])
        print(name, count_predicted(records))
        print(" text:", records[0]["text"])
        print(" predicted:", records[0]["predicted"], "label:", records[0]["label"])