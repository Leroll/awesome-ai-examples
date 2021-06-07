from gensim.models.word2vec import KeyedVectors
import numpy as np
from utils import call_time_with_name

# TODO 后续做成一个类,self.word2vec


# word2vec
@call_time_with_name('load_word2vec')
def load_word2vec(path='../nlp_resource/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.txt'):
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

