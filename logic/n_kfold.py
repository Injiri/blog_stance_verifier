import sys
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from logic.feature_modeling import reguting_features,polarity_feature,generate_or_load_feats,hand_features
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