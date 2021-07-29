import pandas as pd
import torch

from utils import log
from utils import time_cost


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


class SimilarityDataset:
    """
    dataset_columns = ['sentence1', 'sentence2', 'label']
    """
    def __init__(self, name='bq_corpus', logger=print):
        self.logger = logger
        self.corpus_func = {
            'bq_corpus': self.get_bq_corpus
        }
        self.train = None
        self.val = None
        self.test = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.corpus_func[name]()  # 获取数据集
        self.create_dataset_loader()  # 建立dataloader
        self.describe_dataset()

    @time_cost
    def get_bq_corpus(self):
        self.logger('start to read BQ_corpus..')
        self.train = pd.read_csv('./data/bq_corpus/train.tsv')
        self.val = pd.read_csv('./data/bq_corpus/dev.csv')
        self.test = pd.read_csv('./data/bq_corpus/test.csv')

    def describe_dataset(self):
        print('\nDescribe:')
        for i in [self.train, self.val, self.test]:
            print(i.shape)

    @time_cost
    def create_dataset_loader(self):
        self.logger('creating data loader')

        def get_dataset_loader(df):
            q1 = df.sentence1.tolist()
            q2 = df.sentence2.tolist()
            y = df.label.tolist()

            data_loader = torch.utils.data.DataLoader(list(zip(q1, q2, y)),
                                                      batch_size=32,
                                                      shuffle=True)
            return data_loader

        self.train_loader = get_dataset_loader(self.train)
        self.val_loader = get_dataset_loader(self.val)
        self.test_loader = get_dataset_loader(self.test)


if __name__ == '__main__':
    bp_corpus = SimilarityDataset(logger=log)

    # TODO 数据集的读取还有点问题,后续需要验证 dataloader是否跑通.
