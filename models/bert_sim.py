import torch
from torch import nn

from pretrain_train_base_model import PretrainBasedModels

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
        if mode in supported_modes:
            self.mode = mode
        else:
            raise ValueError(f'mode must be in: {supported_modes}')

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