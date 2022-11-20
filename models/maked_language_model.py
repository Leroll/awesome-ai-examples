from collections import OrderedDict
import torch
from torch import nn
import numpy as np
import random
from utils.utils import *


class MaskedLanguageModel(nn.Module):
    """mlm, pet_mlm 均可
    """
    def __init__(self,
                 name: str,
                 device: str,
                 pretrain_model
                 ):
        """
        args：
            name： 模型名字
            device： 'cpu' or 'cuda:0'
            pretrain_model: pretrain_model, 抱抱脸格式
        """
        super().__init__()
        self.name = name
        self.device = device
        self.pretrain_model = pretrain_model
        self.final_part = self.__init_layers()

        self.to(self.device)  # 需要放在最后，等所有weight都初始化后再更改device

    def __init_layers(self):
        hidden_size = self.pretrain_model.config.hidden_size
        embedding_size = self.pretrain_model.embeddings.word_embeddings.num_embeddings
        final_part = nn.Sequential(OrderedDict([
            ('final_Linear',
             nn.Linear(hidden_size, hidden_size, bias=True)),
            ('final_layernorm',
             nn.LayerNorm(hidden_size, eps=1e-12)),
            ('final_embedding',
             nn.Linear(hidden_size, embedding_size, bias=False))
        ]))
        return final_part

    def forward(self, x):
        """x为pair对tokenize 好之后的token_id
        """
        x = self.pretrain_model(**x)[0]  # hidden
        y_pre = self.final_part(x)

        return y_pre


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












