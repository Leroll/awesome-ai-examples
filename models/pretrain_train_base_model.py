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

    def save(self, prefix=''):
        path = f'{prefix}_{self.name}.model'
        torch.save(self.state_dict(), path)

    def load(self, prefix=''):
        path = f'{prefix}_{self.name}.model'
        self.load_state_dict(torch.load(path))
