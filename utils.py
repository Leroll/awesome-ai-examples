from time import time
import torch
import random
import numpy as np
import os
from pypinyin import lazy_pinyin
import jieba


def time_cost(func):
    def wrapper(*arg, **kargs):
        t0 = time()
        res = func(*arg, **kargs)
        t1 = time()
        print(f'[{func.__name__}] cost {t1 - t0:.2f}s')
        return res
    return wrapper


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


def set_deterministic(seed=42):
    """
    让pytorch训练过程可复现，固定各种随机化的操作。
    """
    # seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # 禁止hash随机化
    os.environ['PYTHONHASHSEED'] = str(seed)

    # https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn那边，使用相同的算法，而不是经过benchmark后在几个中选取最快的。
    torch.backends.cudnn.benchmark = False

    # torch内部一些方法使用确定型的算法，具体list见函数文档
    # torch.use_deterministic_algorithms(True)

    # 当上述cudnn使用同一算法时，有可能算法本身不是确定性的，因此需要下述设定
    # 但是该设定已经被上面的设定包含了。
    torch.backends.cudnn.deterministic = True

    # dataloader在多进程时也会有reproducibility的问题
    # 这部分暂时不涉及。


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


# TODO 腾讯word vector， 后续考虑放到别的地方
from gensim.models.word2vec import KeyedVectors


def load_word2vec(path=None):
    if path is None:
        path = '../nlp_resource/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.txt'
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