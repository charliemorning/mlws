import logging

import pandas as pd
from tqdm import tqdm

from util.eda import *


tqdm.pandas()


def load_labelled_data(path):
    """
    header: id,class_label,content
    :param path:
    :return:
    """
    df = pd.read_csv(path)

    df["content_seq"] = df["content"].progress_apply(jieba_tokenize)
    # feature_selection(df["content_seq"], df["class_label"])
    seq_stat = SequenceStatistics(df["content_seq"])
    logging.info(seq_stat)
    return df


def load_unlabelled_data(path):
    """
    header: id,content
    :param path:
    :return:
    """
    df = pd.read_csv(path)


def load_test_data(path):
    """
    header: id,content
    :param path:
    :return:
    """
    df = pd.read_csv(path)


if __name__ == '__main__':
    load_labelled_data(r'C:\Users\charlie\developer\data\nlp\compete\ccf\2020\面向数据安全治理的数据内容智能发现与分级分类\labeled_data.csv')