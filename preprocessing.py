import re
from pathlib import Path
import contractions
import json
import emoji


DEFAULT_EMOTICONS = {
    ":)": "happy_face",
    ":-)": "happy_face",
    ":(": "sad_face",
    ":-(": "sad_face",
    ":D": "laughing_face",
    ":-D": "laughing_face",
    ";)": "winking_face",
    ":P": "playful_face",
}


def preprocessing_text(text: str):
    # clean order: numbers, contractions, url/user, emoji, ascii, slang, lower, stopwords
    text_remove_numbers = remove_numbers(text)
    text_expand_negations_library = expand_negations_library(
        text_remove_numbers
    )
    text_without_space = text_expand_negations_library.strip()
    text_with_tokens = remplacement_with_tokens(text_without_space)
    text_convert_emojis = convert_emojis_and_emoticons(text_with_tokens)
    text_remove_non_ascii = remove_non_ascii(text_convert_emojis)
    text_expand_slang = expand_slang(text_remove_non_ascii)
    text_normalized = text_expand_slang.lower()
    text_normalized = re.sub(r"[^\w\s]", " ", text_normalized)
    test_remove_stopwords = remove_stopwords(text_normalized)

    word_list = test_remove_stopwords

    return word_list


def remplacement_with_tokens(text: str):
    # replace links and @mentions before punctuation is removed
    url_patter = r"https?://\S+|www\.\S+|(?<!\w)[a-zA-Z0-9-]+(?:\.[a-zA-Z]{2,})+(?:/\S*)?"
    user_patter = r"@\w+"

    normalized_text = re.sub(url_patter, " URL ", text)
    normalized_text = re.sub(user_patter, " USER ", normalized_text)

    return normalized_text


def remove_non_ascii(text):
    # drop characters outside basic ascii
    non_ascii = r"[^\x00-\x7F]+"
    return re.sub(non_ascii, "", text)


def remove_numbers(text):
    return re.sub(r"\d+", "", text)


def expand_negations_library(text):
    # don't -> do not, so negation is not lost
    return contractions.fix(text)


def load_slang_dict(filepath="data/dictionaries/slang_dict.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def expand_slang(text):
    slang_dict = load_slang_dict()
    words = text.split()
    expanded_words = []
    for word in words:
        clean_word = word.lower()

        if clean_word in slang_dict:
            expanded_words.append(slang_dict[clean_word])
        else:
            expanded_words.append(word)
    return " ".join(expanded_words)


def load_van_stoplist(filepath="data/dictionaries/van_stoplist.txt"):
    with open(filepath, "r", encoding="utf-8") as f:
        # load each word into a set
        return set(line.strip().lower() for line in f if line.strip())


def remove_stopwords(text):
    # keep negation words for the classifier
    stop_words_set = load_van_stoplist() - {"not", "no", "never", "but"}
    words = text.split()
    filtered_words = [
        word for word in words if word.lower() not in stop_words_set
    ]
    return " ".join(filtered_words)


def load_emoticon_dict(filepath="data/dictionaries/emoticon_dict.json"):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return DEFAULT_EMOTICONS


def convert_emojis_and_emoticons(text, emoticon_dict=None):
    # :) and unicode emoji become words
    if emoticon_dict is None:
        emoticon_dict = load_emoticon_dict()

    for emo, text_rep in emoticon_dict.items():
        text = text.replace(emo, f" {text_rep} ")

    text = emoji.demojize(text, delimiters=(" ", " "))

    return text