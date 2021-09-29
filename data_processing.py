import pandas as pd
import numpy as np
import random
from multiprocessing import Pool
import sys
import torch
import jieba

from utils import *


class DataProcessor:
    """从源文件读取数据到Dataloader过程中的各种工具函数.
    """

    def __init__(self, tokenizer=None, logger=print):
        """
        args:
            tokenizer: 来自抱抱脸的tokenizer， mask的时候需要
        """
        self.logger = logger
        self.tokenizer = tokenizer
        self.vocab = self.tokenizer.vocab

    ############################################
    # 文件读取
    ############################################
    @time_cost
    def read_data(self, mode, name, path, sep, encoder='utf-8', has_index=False):
        """
        读取数据,返回 list形式的数据
        处理如下类型数据集
        [q1 q2 label]

        [idx q1 q2 label]

        mode: 读取数据的方式,
              readline
              pandas
        """
        self.logger(f'-' * 42)
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
    def _read_data_by_readline(path, sep, encoder='utf-8', has_index=False):
        data = []
        with open(path, encoding=encoder) as f:
            line = f.readline()
            while line:
                try:
                    # 预处理
                    line = line.strip()
                    line = line.replace('\ufeff', '')

                    if has_index:
                        idx, q1, q2, label = line.split(sep)
                    else:
                        q1, q2, label = line.split(sep)
                    data.append([q1, q2, label])

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

    ############################################
    # preprocessing
    ############################################
    def preprocessing(self, text: str):
        """单句处理

        1. 字母小写化
        2. 无意义字符删除
        3. 近形字的改写
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
                '歀': '款', '廯': '癣'
            }
            if i in char_correct:
                i = char_correct[i]

            # nonsense letter
            if i in [' ', ' ', '　', '　', ' ', ' ',
                     chr(8198), chr(65039), chr(8237), chr(8236),  # 一串打出来都是空格
                     chr(8419),
                     '\u200d', '\x08', '', '',
                     '∨', '乛', '∵', chr(8198), ]:
                continue

            # UNK
            if self.vocab is None:
                raise Exception('vocab is none, you need to set it before running this func')

            if i not in self.vocab:
                self.logger(text, '|', ord(i), '|', i)
                i = '[UNK]'  # 这个UNK后面的 tokenizer可以处理

            res += i

        return res

    def preprocess_similarity_data(self, data):
        """
        预处理形如 [[q1_1,q2_1,label_1], [q1_2,q2_2,label_2]] 的数据
        """
        q1, q2, label = list(zip(*data))
        q1 = [self.preprocessing(q) for q in q1]
        q2 = [self.preprocessing(q) for q in q2]
        res = list(zip(q1, q2, label))
        return res

    @staticmethod
    def word_segmentation(s: str):
        """
        分词

        #TODO 后续加入 hanlp 分词对比
        """
        seg_s = jieba.lcut(s)
        return seg_s

    #################################################
    # multiprocessing
    #################################################
    @time_cost
    def multi(self, work_num, func, data):
        """
        目前函数输入只有 data
        """
        per_lenght = len(data) // work_num

        p = Pool()
        p_res = []
        for i in range(work_num):
            begin = per_lenght * i
            end = per_lenght * (i + 1) if i != (work_num - 1) else len(data)
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

    ##################################################
    # mask data
    ##################################################
    def get_masked_text(self, text):
        """生成masked后的输入inputs和对应的输出标签 labels
        为了利用tokenizer, labels也生成文字序列, 不预测的部分用[PAD]代替
        """
        if self.vocab is None:
            raise Exception('vocab is none, you need to set it before running this func')

        inputs = []
        labels = []

        text = self.tokenizer.tokenize(text)  # string 分离个tokens
        # 获取 masked tokens
        masked_num = max(1, round(len(text) * 0.15))
        masked_tokens = random.sample(text, masked_num)

        for i in text:
            if i in masked_tokens:
                r = np.random.random()
                if r <= 0.8:
                    inputs.append('[MASK]')
                    labels.append(i)
                elif r <= 0.9:
                    inputs.append(np.random.choice(list(text)))
                    labels.append(i)
                else:
                    inputs.append(i)
                    labels.append(i)
                masked_tokens.remove(i)
            else:
                inputs.append(i)
                labels.append('[PAD]')

        # 转回字符串, 字符之间会多出空格，但是没关系tokenizer后续会忽略
        inputs = self.tokenizer.convert_tokens_to_string(inputs)
        labels = self.tokenizer.convert_tokens_to_string(labels)
        return [inputs, labels]

    def get_masked_data(self, data):
        """
        data = [q1,q2,q3,...]

        return: [[input_1, label_1],[input_2, label_2],...]
        """
        res = [self.get_masked_text(i) for i in data]

        self.logger('Inputs & labels:')
        for i in range(5):
            self.logger(res[i])

        return res
