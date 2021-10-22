import torch
from torch import nn
from utils import *


class BertSim(nn.Module):
    """用于基本文本相似度的fine-tune
    在bert输出中支持三种选择：
        cls
        hidden-mean
        hidden-mean，忽略mask的部分
    """
    def __init__(self,
                 name: str,
                 device: str,
                 pretrain_model,
                 mode: str):
        """
        args：
            mode：bert输出的选择
                'cls':
                'hidden_mean': hidden 输出整体mean
                'hidden_mean_mask': hidden 输出去掉 padding 影响后 mean
        """
        supported_modes = ['cls', 'hidden_mean', 'hidden_mean_mask']
        if mode in supported_modes:
            self.mode = mode
        else:
            raise ValueError(f'mode must be in: {supported_modes}')

        super().__init__()
        self.name = name
        self.device = device
        self.pretrain_model = pretrain_model
        self.fine_tune = self.__init_layers()
        self.to(self.device)  # 需要放在最后，等所有weight都初始化后再更改device

    def __init_layers(self):
        hidden_size = self.pretrain_model.config.hidden_size
        fine_tune = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        return fine_tune

    def _through_bert_then_mean(self, inputs, mean_mode='hidden_mean_mask'):
        """
        through bert & get query representation
        hidden output mean

        args:
            token :  tokenizer output, dict
            mean_mode:
                'hidden_mean', 全部 hidden output 直接求 mean
                'hidden_mean_mask', 去掉 padding 部分的影响，剩余的token进行mean
        """
        supported = ['hidden_mean', 'hidden_mean_mask']

        q = self.pretrain_model(**inputs)[0]  # 0是hidden
        if mean_mode == 'hidden_mean':
            q = torch.mean(q, dim=1)
            return q

        if mean_mode == 'hidden_mean_mask':
            attention_mask = torch.unsqueeze(inputs['attention_mask'], 2)
            q = (attention_mask * q).sum(1) / attention_mask.sum(1)
            return q

        raise ValueError(f'The value of mean_mode must be one of the following:{supported}')

    def forward(self, inputs):
        """
        args:
            inputs: tokenizer之后的字典
        """
        if self.mode == 'cls':
            q = self.pretrain_model(**inputs)[1]
        elif self.mode == 'hidden_mean':
            q = self._through_bert_then_mean(inputs, 'hidden_mean')
        elif self.mode == 'hidden_mean_mask':
            q = self._through_bert_then_mean(inputs, 'hidden_mean_mask')
        else:
            raise ValueError('unexpected value of self.mode')

        q = self.fine_tune(q)
        return q


class TransSimData(object):
    """从初步的匹配数据集中转换成model能使用的数据
    [[q1, q2, label], ... ]
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

    def trans_data(self, batch: list):
        """把初步数据集转换 model 接收的数据格式

        原始数据:
             [[q1_1, q2_1, label_1], ... ]

        return:
            x: dict, 和tokenizer返回的一致
                {'input_ids': , 'token_type_ids': , 'attention_mask':}
        """
        q1, q2, label = list(zip(*batch))
        inputs = self.get_token_from_pair(q1, q2, is_split_into_words=False)
        label = torch.tensor(label).to(self.device)

        return inputs, label
