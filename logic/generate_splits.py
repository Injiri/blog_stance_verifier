import random
import os
from collections import defaultdict

training = 0.8


def gen_holdout_split(datasets, training, base_dir="/splits_data"):
    r = random.Random()
    r.seed(1489215)

    article_ids = list(datasets.articles.keys())
    r.shuffle(article_ids)

    training_ids = article_ids[:int(training * len(article_ids))]
    hold_out_ids = article_ids[int(training * len(article_ids)):]

    # write split bodyids to file for later use
    with open(base_dir + "/" + "training_ids.txt", "w+") as f:
        f.write("\n".join([str(id) for id in training_ids]))

    with open(base_dir + "/" + "hold_out_ids.txt", "w+") as f:
        f.write("\n".join([str(id) for id in hold_out_ids]))


def read_ids(file, base):
    ids_array = []
    with open(base + "/" + file, 'r') as f:
        for line in f:
            ids_array.append(int(line))
        return ids_array


def kfold_split(datasets, training=0.8, n_folds=10, base_dir="splits_data"):
    if not (os.path.exists(base_dir + "/" + "training_ids.txt")
            and os.path.exists(base_dir + "/" + "hold_out_ids.txt")):
        gen_holdout_split(datasets, training, base_dir)

    training_ids = read_ids("training_ids.txt", base_dir)
    holdout_ids = read_ids("hold_out_ids.txt", base_dir)

    folds_array = []
    for f in range(n_folds):
        folds_array.append(training_ids[int(f * len(training_ids) / n_folds):
                                        int((f + 1) * len(training_ids) / n_folds)])

    return folds_array, holdout_ids


def get_stances_4_folds(datasets, folds, hold_out):
    stances_folds = defaultdict(list)
    stances_hold_out = []
    for stance in datasets.stances:
        if stance['Body ID'] in hold_out:
            stances_hold_out.append(stance)
        else:
            fold_id = 0
            for fold in folds:
                if stance["Body ID"] in fold:
                    stances_folds[fold_id].append(stance)
                fold_id += 1
    return stances_folds, stances_hold_out
