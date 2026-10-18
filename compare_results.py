from classify_tweets import classify_records, count_predicted
from bigram_model import train_datasets
from dataset_loader import count_labels


def compare_records(records):
    # predicted is the model, label is the original csv tag
    comparison = {
        "original": count_labels(records),
        "predicted": count_predicted(records),
        "same": 0,
        "different": 0,
    }

    for record in records:
        if record["predicted"] == record["label"]:
            comparison["same"] += 1
        else:
            comparison["different"] += 1

    comparison["total"] = len(records)
    comparison["accuracy"] = comparison["same"] / len(records)
    return comparison


if __name__ == "__main__":
    datasets, models = train_datasets()
    for name, records in datasets.items():
        classify_records(records, models[name])
        comparison = compare_records(records)
        print(name)
        print(" original:", comparison["original"])
        print(" predicted:", comparison["predicted"])
        print(" same:", comparison["same"], "different:", comparison["different"])
        print(" accuracy:", round(comparison["accuracy"], 4))