from time import time
import torch
import random
import numpy as np
import os
from pypinyin import lazy_pinyin
import jieba
from gensim.models.word2vec import KeyedVectors


def reverse_dict(x: dict):
    """交换字典的 key-value, 得到 value-key 的新字典
    需保证value无重复项
    """
    if isinstance(x, dict):
        k, v = list(zip(*list(x.items())))
        x_reverse = {}
        for i in range(len(k)):
            x_reverse[v[i]] = k[i]  # k-v 反转字典
        return x_reverse
    else:
        raise TypeError('arg needs to be dict')


class ModelConfig(dict):
    """config类
    """
    def __init__(self, name):
        super().__init__()
        self['name'] = name

    def __getattr__(self, item):
        if item in self:
            return self[item]
        else:
            raise AttributeError(f'No such attribute: {item}')

    def __setattr__(self, key, value):
        self[key] = value


def compare_pinyi(s1: str, s2: str):
    """
    对比两句话是否拼音完全一致。
    """
    s1_pinyin = ""
    s2_pinyin = ""
    for w in jieba.cut(s1):
        s1_pinyin += ''.join(lazy_pinyin(w))
    for w in jieba.cut(s2):
        s2_pinyin += ''.join(lazy_pinyin(w))
    return s1_pinyin == s2_pinyin


# -----------------------------------
# TODO 下面是腾讯的word vector， 后续考虑放到别的地方
def load_word2vec(path=None):
    if path is None:
        path = '../nlp_resource/Tencent_AILab_ChineseEmbedding/\
                Tencent_AILab_ChineseEmbedding.txt'
    word2vec = KeyedVectors.load_word2vec_format(path,
                                                 binary=False)
    return word2vec


def sentence_vector_by_word2vec(q):
    """
    average word2vec
    """
    word2vec = load_word2vec()  # TODO
    res = np.zeros(200)
    cnt = 0
    for w in q:
        res += word2vec[w] if w in word2vec else 0
        cnt += 1
    res /= cnt
    return res
