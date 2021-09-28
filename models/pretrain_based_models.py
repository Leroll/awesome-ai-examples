from collections import OrderedDict

import torch
from torch import nn


class PretrainBasedModels(nn.Module):
    """使用预训练模型的基础模型类

    tokenizer & pretrain_model 来自抱抱脸
    """

    def __init__(self,
                 name: str,
                 device: str,
                 tokenizer,
                 pretrain_model,
                 max_len: int,
                 **kwargs):
        """
        args：
            name： 模型名字
            device： 'cpu' or 'cuda:0'
            tokenizer: 与pretrain_model 匹配的 tokenizer
            pretrain_model: pretrain_model, 抱抱脸格式
            max_len: tokenize 时句子最大长度
        """
        super().__init__()
        self.name = name
        self.device = device
        self.tokenizer = tokenizer
        self.pretrain_model = pretrain_model
        self.max_len = max_len

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

    def _through_bert_then_mean(self, token, mean_mode='attention_mask'):
        """
        through bert & get query representation
        hidden output mean

        args:
            token :  tokenizer output, dict
            mean_mode:
                'mean', 全部 hidden output 直接求 mean
                'attention_mask', 去掉 padding 部分的影响，剩余的token进行mean
        """
        supported = ['mean', 'attention_mask']

        q = self.pretrain_model(**token)[0]  # 0是hidden
        if mean_mode == 'mean':
            q = torch.mean(q, dim=1)
            return q
        if mean_mode == 'attention_mask':
            attention_mask = torch.unsqueeze(token['attention_mask'], 2)
            q = (attention_mask * q).sum(1) / attention_mask.sum(1)
            return q

        raise ValueError(f'The value of mean_mode must be one of the following:{supported}')

    def save(self, path=None):
        if path is None:
            path = './' + self.name + '.model'
        torch.save(self.state_dict(), path)

    def load(self, path=None):
        if path is None:
            path = './' + self.name + '.model'
        self.load_state_dict(torch.load(path))


class SentenceBert(PretrainBasedModels):
    """
    args:
        pretrain_model: 需要是bert
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        hidden_size = self.pretrain_model.config.hidden_size
        self.linear = nn.Linear(hidden_size * 3, 2)

    def forward(self, q1, q2):
        q = []
        for temp_q in [q1, q2]:
            temp_q = self.get_token_from_single(temp_q, is_split_into_words=False)
            temp_q = self._through_bert_then_mean(temp_q, mean_mode='attention_mask')
            q.append(temp_q)

        diff = torch.abs(q[0] - q[1])
        h = torch.cat((q[0], q[1], diff), dim=-1)
        h = self.linear(h)
        return h


class BertSim(PretrainBasedModels):
    """
    用于基本文本相似度的fine-tune
    """
    def __init__(self,
                 mode: str,
                 **kwargs):
        """
        args：
            mode：bert输出的选择
                'cls':
                'hidden_mean': hidden 输出整体mean
                'hidden_mean_mask': hidden 输出去掉 padding 影响后 mean
        """
        super().__init__(**kwargs)

        supported_modes = ['cls', 'hidden_mean', 'hidden_mean_mask']
        if mode in self.supported_modes:
            self.mode = mode
        else:
            raise ValueError(f'mode must be in: {self.supported_modes}')

        hidden_size = self.pretrain_model.config.hidden_size
        self.fine_tune = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.linear = nn.Linear(hidden_size, 2)

    def forward(self, q1, q2):
        token = self.get_token_from_pair(q1, q2, is_split_into_words=False)
        if self.mode == 'cls':
            q = self.pretrain_model(**token)[1]
        elif self.mode == 'hidden_mean':
            q = self._through_bert_then_mean(token, 'mean')
        elif self.mode == 'hidden_mean_mask':
            q = self._through_bert_then_mean(token, 'attention_mask')
        else:
            raise ValueError('unexpected value of self.mode')

        q = self.finetune(q)
        return q


class MlmBert(PretrainBasedModels):
    """用于 mlm 训练
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        hidden_size = self.pretrain_model.config.hidden_size
        embedding_size = self.pretrain_model.embeddings.word_embeddings.num_embeddings
        self.final_part = nn.Sequential(OrderedDict([
            ('final_Linear',
             nn.Linear(hidden_size, hidden_size, bias=True)),
            ('final_layernorm',
             nn.LayerNorm(hidden_size, eps=1e-12)),
            ('final_embedding',
             nn.Linear(hidden_size, embedding_size, bias=False))
        ]))

        # 因为emb的w本身就是v*d的, 这里不需要转置
        embedding_p = [p for p in self.pretrain_model.embeddings.word_embeddings.parameters()][0]
        self.final_part.final_embedding.weight.data = embedding_p

    def forward(self, x):
        token = self.get_token_from_single(x, is_split_into_words=True)
        x = self.pretrain_model(**token)[0]
        y_pre = self.final_part(x)

        return y_pre

    def get_y_mask(self, y):
        """
        like attention_mask, get y_mask from y_token
        """
        y_mask = []
        for i in y:
            temp_mask = []
            for j in i:
                m = 0 if j in [0, 101, 102] else 1
                temp_mask.append(m)
            y_mask.append(temp_mask)

        y_mask = torch.tensor(y_mask).to(self.device)
        return y_mask




