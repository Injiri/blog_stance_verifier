import sys
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from logic.feature_modeling import refuting_features,polarity_feature,generate_or_load_feats,hand_features
from logic.feature_modeling import word_overlap_features
from util_files.datasets import Datasets
from util_files.generate_splits import kfold_split,get_stances_4_folds
from  util_files.score import  report_score, LABELS, submit_score
from util_files.sys_driver import parameter_parser, versioning

def generate_features(stances, dataset, name ):

    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body id']])

    x_overlap = generate_or_load_feats(word_overlap_features, h, b, "feaures/refuting."+name+".npy")
    x_refuting = generate_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    x_polarity = generate_or_load_feats()

    X = np.c_
    return  X,y

#Generate folds
if __name__ == "__main__":
    versioning()
    #load the traing dataset
    parameter_parser()
    my_dataset = Datasets()
    folds,hold_out = kfold_split(my_dataset,n_folds=10)
    fold_stances, hold_out_stances = get_stances_4_folds(my_dataset,folds,hold_out)

    demo_dateset = Datasets("Demo /test")
    X_demo, Y_demo = generate_features(demo_dateset.stances, demo_dateset, "demo")

    Xs = dict()
    Ys = dict()

    #populat all features
    X_holdout,y_holdout = generate_features(hold_out_stances, my_dataset,"holder")
    for fold in fold_stances:
        Xs[fold],Ys[fold] = generate_features(fold_stances[fold] , demo_dateset ,str(fold))

    best_score = 0
    best_fold = None

    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        Y_train = np.hstack(tuple([Ys[i] for i in ids]))

        x_demo = Xs[fold]
        Y_demo = Ys[fold]

        classifier = GradientBoostingClassifier(n_estimaters=200, random_state=14128, verbose=True)
        classifier.fit(X_train, Y_train)

        predicted_result = [LABELS[int(n)] for n in classifier.predict(X_demo)]
        actual_result = [LABELS[int(n)] for n in Y_demo ]

        fold_score, _ = submit_score(actual_result, predicted_result)
        max_fold_score, _ =submit_score(actual_result, actual_result);

        weigthted_score = fold_score/max_fold_score

        print(str(fold) + "is Fold score initialy was" + str(weigthted_score))
        if weigthted_score > best_score:
            best_score = weigthted_score
            best_score = classifier


    #report the  final best_score
    predicted = [LABELS[int (n)] for n in best_fold.predict(X_holdout)]
    actual_result = [LABELS(int (n)) for n in Y_demo]

    report_score(actual_result, predicted)