import pandas as pd
import numpy as np
import random
from multiprocessing import Pool
import sys
import torch

from utils import *


class DataProcessor:
    """ reader 和 一些相关函数.
    形成初步的数据集
    """

    def __init__(self, logger=print):
        self.logger = logger

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
    def _read_data_by_readline(path, sep, encoder='utf-8', has_index=False, has_label=True):
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

    #################################################
    # multiprocessing
    #################################################
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

    ##################################################
    # split data
    ##################################################
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


class Masker(object):
    """mask 相关的数据处理类
    用于从将初步的数据转换成与模型匹配的数据格式
    """

    def __init__(self, tokenizer, max_len, device, logger=print):
        """
        args:
            tokenizer: 来自抱抱脸
        """
        self.logger = logger
        self.max_len = max_len
        self.device = device
        self.tokenizer = tokenizer
        self.vocab = tokenizer.vocab
        self.vocab_inverse = reverse_dict(self.vocab)

        # 一些关键token
        self.mask_id = self.vocab['[MASK]']
        self.cls_id = self.vocab['[CLS]']
        self.sep_id = self.vocab['[SEP]']
        self.pad_id = self.vocab['[PAD]']

    def get_masked_text(self, input_ids: list, exclude_ids: list):
        """将单样本的input_ids 转换为 masked_input_ids,同时生成对应的 masked_input_labels
        未mask的label设置为0。注意是元素是token_id，而非token

        args:
            input_ids: 经过抱抱脸tokenizer之后的input_ids, 包含[CLS][SEP][PAD]等字符
            exclude_ids: 不需要mask的id，比如101:[CLS], 102:[SEP], 0:[PAD]
        return:
            masked_input_ids: 除了[CLS],[PAD],[SEP]等词以外，进行mask 替换
            masked_labels: 与输入等长,对应的label，只有mask对应的字符有值，其余为0，包括[CLS],[SEP]
            mask_flag: 用于计算loss时，筛选出需要计算loss的项。类似tokenizer 的attention_mask

        """
        masked_input_ids = []
        masked_labels = []

        # 获取 masked tokens
        masked_index_pool = [idx for idx, i in enumerate(input_ids) if i not in exclude_ids]  # 候选池的索引
        masked_num = max(1, round(len(masked_index_pool) * 0.15))
        masked_index = random.sample(masked_index_pool, masked_num)

        for idx, i in enumerate(input_ids):
            if idx in masked_index:
                r = np.random.random()
                if r <= 0.8:
                    masked_input_ids.append(self.mask_id)
                elif r <= 0.9:
                    replace_id = input_ids[np.random.choice(list(masked_index_pool))]
                    masked_input_ids.append(replace_id)
                else:
                    masked_input_ids.append(i)
                masked_labels.append(i)
            else:
                masked_input_ids.append(i)
                masked_labels.append(self.pad_id)

        mask_flag = [int(i > 0) for i in masked_labels]
        return [masked_input_ids, masked_labels, mask_flag]

    def get_masked_data(self, inputs: dict, exclude_ids: list):
        """对batch进行mask

        args:
            inputs: 抱抱脸tokenizer 的返回值

        return:
            x: dict, 和tokenizer返回的一致
                {'input_ids': , 'token_type_ids': , 'attention_mask':}
            y: torch.tensor, masked_label_token_id, 非mask 的token id 为 0
            y_mask: torch.tensor, bool值的list，指示哪些字符用于计算loss
        """
        mask_res = [self.get_masked_text(i, exclude_ids) for i in inputs['input_ids']]
        masked_input_ids, masked_labels, mask_flags = tuple(zip(*mask_res))

        inputs['input_ids'] = torch.tensor(masked_input_ids).to(self.device)
        x = inputs
        y = torch.tensor(masked_labels).to(self.device)
        y_mask = torch.tensor(mask_flags).to(self.device)
        return x, y, y_mask

    def get_token_from_single(self, q, is_split_into_words=False):
        tokens = self.tokenizer(text=q,
                                padding='longest',
                                truncation=True,
                                max_length=self.max_len,
                                is_split_into_words=is_split_into_words,  # True:输入为str，False:切分好的list
                                return_tensors='pt'
                                ).to(self.device)
        return tokens

    def get_token_from_pair(self, q1, q2, is_split_into_words=False):
        tokens = self.tokenizer(text=q1,
                                text_pair=q2,
                                padding='longest',
                                truncation=True,
                                max_length=self.max_len,
                                is_split_into_words=is_split_into_words,  # True:输入为str，False:切分好的list
                                return_tensors='pt'
                                ).to(self.device)
        return tokens

    def trans_to_masked_data(self, batch: list):
        """把初步数据集转换为 masked 数据集

        原始数据:
            [q1,q2, ... ]

        return:
            x: dict, 和tokenizer返回的一致
                {'input_ids': , 'token_type_ids': , 'attention_mask':}
            y: mask label token id, 非mask 的token id 为 0，即 [PAD]对应的id
            y_mask: bool值的list，指示哪些字符用于计算loss
        """
        q = batch
        inputs = self.get_token_from_single(q, is_split_into_words=False)
        exclude_ids = [self.cls_id, self.sep_id, self.pad_id]

        x, y, y_mask = self.get_masked_data(inputs, exclude_ids)
        return x, y, y_mask

    def trans_to_pet_masked_data(self, batch: list):
        """对原始的文本数据进行处理，得到处理好的id和label

        train:
            0. label的类别分别使用'是','否','[PAD]',来对应1，0和测试集的labels
            1. 注意 mask 之后，其实有了两种不同的label，
               a. mask_label，就是对应mask掉字符的label，在本程序里仍然是string
               b. 原本的相似 label
            2. 因为想要根据每个batch的最大长度来打padding，所以不能对全部数据处理完之后，再生成dataloader
            3. 直接根据原始的数据集，生成dataloader，然后配合collate_fn在每次循环的时候处理
            4. 由于mask也是在这里做的，所以就意味着, 每一次mask的结果都不一样，可以直接多次循环
            5. 之前的做法是先对token，mask，再batch，再tokenizer得到输入，这样就会tokenize 2次
               从而导致'##00'这种 word-piece 被再次切分成为'#'，'#'，'00'，从而导致inputs和label的长度
               不一致，出现问题。
               现在是更改成了先全部tokenizer成为 token_id之后，再去做mask。

        test:
            1. test阶段文本就不需要做新的label了，直接输入tokenize一下

        args:
            batch: [[q1,q2,label],[q1,q2,label],...]

        return:
            x: dict, 和tokenizer返回的一致
                {'input_ids': , 'token_type_ids': , 'attention_mask':}
            y: cls的值和那些mask掉的字符的id
            y_mask: bool值的list，指示哪些字符用于计算loss
        """
        if len(batch[0]) == 3:  # training
            q1, q2, label = list(zip(*batch))

            inputs = self.get_token_from_pair(q1, q2)
            exclude_ids = [self.cls_id, self.sep_id, self.pad_id]
            x, y, y_mask = self.get_masked_data(inputs, exclude_ids)

            # TODO, 对于mask的比例，先每句话都mask，后续要看看pet的原论文，里面具体mask的比例
            # 更新 y 的 cls label
            label_map = {
                1: '是', 0: '否', -1: '[PAD]'
            }
            for idx, i in enumerate(label):
                cls_id = self.vocab[label_map[i]]
                y[idx][0] = cls_id
                y_mask[idx][0] = 1
            return x, y, y_mask
        else:
            q1, q2 = list(zip(*batch))
            x = self.get_token_from_pair(q1, q2)
            return x, q1, q2  # 加上q1，q2是为了方便组装预测结果
