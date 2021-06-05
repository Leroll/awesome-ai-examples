import jieba


def get_data(name: str):
    """
    read dataset
    """
    if name == 'bq_corpus':
        q1, q2, y = [], [], []
        f = open('./data/bq_corpus/train.tsv', encoding='utf-8')
        line = f.readline()
        while line:
            temp_q1, temp_q2, temp_y = line.strip().split('\t')
            q1.append(temp_q1)
            q2.append(temp_q2)
            y.append(temp_y)
            line = f.readline()
    else:
        raise Exception(f'got wrong dataset name:{name}')
    return q1, q2, y


def word_segmentation(s: str):
    """
    cut sentence

    #TODO 后续加入 hanlp 分词对比
    """
    seg_s = jieba.lcut(s)
    return seg_s
