import pandas as pd
import random
from multiprocessing import Pool
import sys
import torch
from utils import time_cost


class DataProcessor:
    """ reader 和 一些相关函数.
    形成初步的数据集 train, dev, test
    """

    def __init__(self, logger=print):
        self.logger = logger

    @time_cost
    def read_data(self, mode, name, path, sep,
                  encoder='utf-8', has_index=False):
        """
        读取数据,返回 list形式的数据
        处理如下类型数据集
        [q1 q2 label]

        [idx q1 q2 label]

        mode: 读取数据的方式,
              readline
              pandas
        """
        self.logger('-' * 42)
        self.logger(f'start to read: [{name}]...')

        if mode == 'readline':
            data = self._read_data_by_readline(path=path,
                                               sep=sep,
                                               encoder=encoder,
                                               has_index=has_index)
        elif mode == 'pandas':
            data = self._read_data_by_pandas(path=path,
                                             sep=sep,
                                             encoder=encoder)
        else:
            raise Exception('mode的值有误')

        # logs
        self.logger(f'finish reading: [{name}]')
        self.logger('nums:', len(data))
        for i in range(5):
            self.logger(data[i])
        return data

    @staticmethod
    def _read_data_by_readline(path, sep, encoder='utf-8',
                               has_index=False, has_label=True):
        data = []
        with open(path, encoding=encoder) as f:
            line = f.readline()
            while line:
                try:
                    # 预处理
                    line = line.strip()
                    line = line.replace('\ufeff', '')

                    if has_label:
                        if has_index:
                            idx, q1, q2, label = line.split(sep)
                        else:
                            q1, q2, label = line.split(sep)
                        data.append([q1, q2, label])
                    else:
                        if has_index:
                            idx, q1, q2 = line.split(sep)
                        else:
                            q1, q2 = line.split(sep)
                        data.append([q1, q2])

                    line = f.readline()
                except Exception as e:
                    print(f'line: {line}')
                    print('-' * 42)
                    print(e)
                    sys.exit()
        return data

    @staticmethod
    def _read_data_by_pandas(path, sep, encoder='utf-8'):
        data = pd.read_csv(path, sep=sep, encoding=encoder)
        data = data.to_numpy().tolist()
        return data

    @staticmethod
    def create_dataloader(data, batch_size, is_shuffle, collate_fn):
        dataloader = torch.utils.data.DataLoader(data,
                                                 batch_size=batch_size,
                                                 shuffle=is_shuffle,
                                                 collate_fn=collate_fn)
        return dataloader

    @time_cost
    def multi(self, work_num, func, data):
        """
        目前函数输入只有 data
        """
        per_length = len(data) // work_num

        p = Pool()
        p_res = []
        for i in range(work_num):
            begin = per_length * i
            end = per_length * (i + 1) if i != (work_num - 1) else len(data)
            p_res.append(p.apply_async(func, args=(data[begin:end],)))
            self.logger(f'Process:{i} | [{begin}:{end}] ')
        p.close()
        p.join()

        res = []
        for i in range(work_num):
            res.append(p_res[i].get())

        # 拼装起来,外层循环得到单进程小res，内层循环得到j
        res = [j for i in res for j in i]
        return res

    def split_data(self, data: list, dev_num: int):
        """划分 train 和 dev 数据集
        train_num = len(data) - dev_num
        """
        len_data = len(data)
        dev_idx = random.sample(range(len_data), dev_num)
        dev_hash_idx = {i: 0 for i in dev_idx}  # 单纯为了dict的hash性质，0无意义

        dev = [data[i] for i in range(len_data) if i in dev_hash_idx]
        train = [data[i] for i in range(len_data) if i not in dev_hash_idx]
        self.logger(f'data:{len(data)}, train:{len(train)}, dev:{len(dev)}')
        return train, dev


class Preprocessor(object):
    """数据预处理类
    """
    def __init__(self, vocab, logger=print):
        self.vocab = vocab
        self.unk_cnt = 0
        self.logger = logger

    def preprocessing(self, text: str):
        """单句处理

        1. 字母小写化
        2. 无意义字符删除
        3. 近形字的改写
        4. 发现 unk 并记录
        """

        res = ''
        for i in text:
            # 大写字母vocab里面没有
            i = i.lower()

            # punctuation
            if i == '…':
                i = '.'
            elif i in ["'", '‘', '’', '“', '”']:
                i = "'"
            elif i in ['—', '―', '—', '`']:  # 注意'-'互相不一样
                i = ','

            # char
            char_correct = {
                '壋': '增', '笫': '第', '囙': '回',
                '呮': '呗', '嚒': '么', '睌': '晚',
                '谝': '骗', '鍀': '得', '昰': '是',
                '伲': '呢', '肔': '服', '凊': '清',
                '挷': '绑', '亊': '事', '腨': '用',
                '戗': '钱', '玏': '功', '筘': '扣',
                '鈤': '日', '颃': '领', '讠': '之',
                '扥': '在', '螚': '能', '甪': '用',
                '茌': '花', '泝': '没', '牫': '我',
                '孒': '了', '镸': '长', '欹': '款',
                '刭': '到', '幵': '开', '怩': '呢',
                '绐': '给', '弍': '式', '淸': '清',
                '夂': '久', '叧': '另', '徣': '借',
                '冋': '回', '敉': '粒', '埭': '贷',
                '仧': '卡', '頟': '额', '捿': '捷',
                '鳓': '嘞', '䃼': '补', '囯': '国',
                '吿': '告', '鞡': '粒', '疷': '底',
                '歀': '款', '廯': '癣', '仦': '小',
                '佷': '很'
            }
            if i in char_correct:
                i = char_correct[i]

            # nonsense letter
            if i in [' ', ' ', '　', '　', ' ', ' ', '\u200d',
                     '\x08', '', '', '∨', '乛', '∵',
                     chr(8198), chr(8236), chr(8237), chr(8419),
                     chr(65039)]:
                continue

            # UNK
            if i not in self.vocab:
                self.logger(f'{self.unk_cnt:4d} | {text} | {ord(i):5d} | {i}')
                self.unk_cnt += 1

            res += i
        return res

    def preprocess_similarity_data(self, data: list):
        """预处理 pair 类数据

        args:
            data: 形如 [[q1,q2,label], ...]  # train & dev
                  或者 [[q1,q2], ...]  # test
        return:
            与输入形式相同
        """
        if len(data[0]) == 3:
            q1, q2, label = list(zip(*data))
        else:
            q1, q2 = list(zip(*data))

        self.unk_cnt = 0
        q1 = [self.preprocessing(q) for q in q1]
        q2 = [self.preprocessing(q) for q in q2]

        if len(data[0]) == 3:
            res = list(zip(q1, q2, label))
        else:
            res = list(zip(q1, q2))
        return res
