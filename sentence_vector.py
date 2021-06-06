from gensim.models.word2vec import KeyedVectors
import numpy as np

# TODO 后续做成一个类,self.word2vec


# word2vec
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
    num = len(q)
    for w in q:
        res += word2vec[w] if w in word2vec else np.zeros(200)
    res /= num
    return res

