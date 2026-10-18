from dataset_loader import load_datasets
from preprocessing import preprocessing_text


def add_tokens(records):
    # preprocessing_text returns a string, split makes the token list
    for record in records:
        clean_text = preprocessing_text(record["text"])
        record["tokens"] = clean_text.split()
    return records


def prepare_datasets():
    datasets = load_datasets()
    for name in datasets:
        datasets[name] = add_tokens(datasets[name])
    return datasets


if __name__ == "__main__":
    datasets = prepare_datasets()
    for name, records in datasets.items():
        print(name, len(records))
        print(" text:", records[0]["text"])
        print(" tokens:", records[0]["tokens"])