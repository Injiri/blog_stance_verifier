import os
import re
import nltk
import numpy as np
from sklearn import feature_extraction
from tqdm import tqdm

_wnl = nltk.wordNetTokenizer()


def normalize_word(w):
    return _wnl.lemmatize(w).lower()


def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in nltk.word_torkenize(s)]


def clean(s):
    #at this point can perfom a regex to eliminate some parts of the text that arent useful at all by re.findall method
    return " ".join(re.findall(r'\w+', s, flag=re.UNICODE)).lower()


def remove_stopwords(l):
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]


def generate_or_load_feats(feat_fn, headlines, bodies, feature_file):
    if not os.path.isfile(feature_file):
        feats = feat_fn(headlines, bodies)
        np.save(feature_file, feats)

    return np.load(feature_file)


def word_overlap_features(headlines, bodies):
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clearn_headline = clean(headline)
        clean_body = clean(body)
        clean_headline = get_tokenized_lemmas(clean_body)
        featurez = [
            len(set(clean_headline).intersection(clean_body) / float(len(set(clean_headline).union(clean_body))))
        ]
        X.append(featurez)


def refuting_features(headlines, bodies):
    _refuting_words = [
        'fake',
        'not',
        'deny',
        'denies',
        'not',
        'despite',
        'debunk',
        'pranks',
        'false',
        'nope',
        'fraud',
        'bogus',
        'doubt',
        'doubts'

    ]
    # for each word, enumerate its features.
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_headline = get_tokenized_lemmas(clean_headline)
        features = [1 if word in clean_headline else 0 for word in _refuting_words]
        X.append(features)
    return X


def polarity_feature(headlines, bodies):
    _refuting_words = [
        'fake',
        'not',
        'deny',
        'denies',
        'not',
        'despite',
        'debunk',
        'pranks',
        'false',
        'nope',
        'fraud',
        'bogus',
        'doubt',
        'doubts'
    ]

    def calculate_polarity(text):
        tokens = get_tokenized_lemmas(text)
        return sum([t in _refuting_words for t in tokens]) % 2

    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        featurez = []
        featurez.append(calculate_polarity(clean_headline))
        featurez.append(calculate_polarity(clean_body))
        X.append(featurez)
    return np.array(X)


def n_grams(input, n):
    input = input.split(' ')
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def chargrams(input, n):
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def append_char_grams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in chargrams(" ".join(remove_stopwords(text_headline.split())), size)]
    grams_hits = 0
    grams_early_hits = 0
    grams_first_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
    features.append(grams_hits)
    features.append(grams_first_hits)
    return features


def append_n_grams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in n_grams(text_headline, size)]
    grams_hits_count = 0
    body_grams_hit = 0
    for gram in grams:
        if gram in text_headline:
            grams_hits_count += 1
        if gram in text_body[:255]:
            body_grams_hit += 1
    features.append(grams_hits_count)
    features.append(body_grams_hit)
    return features


def handle_features(headlines, bodies):
    def binary_co_occurance(headline, body):
        token_count = 0
        body_token_count = 0
        for headline_token in clean(headline).split(" "):
            token_count += 1
            if headline_token in clean(body)[:255]:
                body_token_count += 1
        return [token_count, body_token_count]

    def binary_occurence_stops(headline, body):
        token_count = 0
        body_token_count = 0
        for headline_token in remove_stopwords(clean(headline).split(" ")):
            if headline_token in clean(body):
                token_count += 1
                body_token_count += 1
        return [token_count, body_token_count]

    def count_grams(headline, body):
        clean_body = clean(body)
        clean_headline = clean(headline)
        features = []
        features = append_char_grams(features, clean_headline, clean_body, 2)
        features = append_char_grams(features, clean_headline, clean_body, 8)
        features = append_char_grams(features, clean_headline, clean_body, 4)
        features = append_char_grams(features, clean_headline, clean_body, 16)
        features = append_n_grams(features, clean_headline, clean_body, 2)
        features = append_n_grams(features, clean_headline, clean_body);
        features = append_n_grams(features, clean_headline, clean_body, 3)
        features = append_n_grams(features, clean_headline, clean_body, 4)
        return features

    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        X.append(binary_co_occurance(headline, body) + binary_occurence_stops(headline, body)
                 + count_grams(headline, body))
    return X
