import csv


polarity_to_label = {
    "0": "negative",
    "2": "neutral",
    "4": "positive",
}


def map_polarity(polarity_code):
    # 0 negative, 2 neutral, 4 positive, same codes in both files
    code = str(polarity_code).strip().strip('"')
    return polarity_to_label[code]


def load_ststd(filepath="data/datasets/testdata.manual.2009.06.14.csv"):
    # no header: polarity, id, date, query, user, text
    records = []

    with open(filepath, "r", encoding="latin-1") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 6 or not row[5].strip():
                continue
            records.append(
                {
                    "text": row[5].strip(),
                    "label": map_polarity(row[0]),
                    "source": "STSTd",
                }
            )

    return records


def load_stsgd(filepath="data/datasets/sts_gold_tweet.csv"):
    # columns id;polarity;tweet
    records = []

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            text = row["tweet"].strip().strip('"')
            if text == "":
                continue
            records.append(
                {
                    "text": text,
                    "label": map_polarity(row["polarity"]),
                    "source": "STSGd",
                }
            )

    return records


def load_datasets():
    datasets = {
        "STSTd": load_ststd(),
        "STSGd": load_stsgd(),
    }
    return datasets


def count_labels(records):
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for record in records:
        counts[record["label"]] += 1
    return counts


if __name__ == "__main__":
    datasets = load_datasets()
    for name, records in datasets.items():
        print(name, len(records), count_labels(records))
        print(" sample:", records[0])